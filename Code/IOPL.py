import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import gurobipy as gp
from gurobipy import GRB
from LP import LP

class IOPL:
    def __init__(self, X, T, Y, M, method='DR', verbose=True, initial_K=None, initial_S=None, mu_1=None, mu_minus1=None, param_grid_mu=None):
                
        # Checks
        if np.amin(X)<-1.0 or np.amax(X)>1.0:
            raise ValueError('Input array X not scaled correctly. Entries not within [-1,1]')
        if ((T!=-1) & (T!=1)).any():
            raise ValueError('Input array T not specified correctly. Entries not within {-1,1}')
        if method not in ['DM','IPS','DR']:
            raise ValueError('Variable method has to be DM, IPS or DR')            
        
        # Data
        self.X=X
        self.Y=Y
        self.T=T
        self.M=M
        self.num_datapoints=X.shape[0]
        self.d=X.shape[1]
        self.method=method
        
        # Settings
        self.epsilon=1e-9
        self.max_iter_for_CG=1000
        self.max_iter_for_branch_and_bound=50
        self.verbose=verbose
        
        # Init Gurobi environment
        self.env=gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.start()
        
        # MISC
        self.mu_1=mu_1
        self.mu_minus1=mu_minus1
        self.param_grid_mu=param_grid_mu
        self.S=None
        self.K=None
        if initial_K is None:
            self.initial_K=[[-np.ones((self.d,1)),np.ones((self.d,1))]]
        else:
            self.initial_K=initial_K.copy()
        if initial_S is None:
            self.initial_S=np.ones((self.num_datapoints,1))
        else:
            self.initial_S=initial_S.copy()
        self.optimal_inds=None
        self.optimal_obj_val_MILP=np.inf
        self.total_CG_iterations=0
        self.total_time_limit_exceeded_CG=0
        
        mu_hat_1, mu_hat_minus1, mu_hat_T = self.estimate_mu()
        e_hat_T = self.estimate_e()
        if self.method=='DM':
            self.psi=self.T.reshape(-1,1)*(mu_hat_minus1-mu_hat_1)
        if self.method=='IPS':
            self.psi=-self.Y/e_hat_T
        if self.method=='DR':
            self.psi=self.T.reshape(-1,1)*(mu_hat_minus1-mu_hat_1)-(self.Y/e_hat_T).reshape(-1,1)+(mu_hat_T.reshape(-1,1)/e_hat_T.reshape(-1,1))
        self.psi/=self.num_datapoints
        
        self.I1=sorted(list(np.where(self.T==1)[0]))
        self.I0=sorted(list(np.where(self.T==-1)[0]))
        self.P=sorted(list(np.where(self.psi>0)[0]))
        self.N=sorted(list(np.where(self.psi<0)[0]))
        if len(list(np.where(self.psi==0.0)[0]))>0:
            raise RuntimeError('psi_'+str(list(np.where(self.psi==0.0)[0])[0])+' is zero!')
        
    def estimate_mu(self):
        
        if self.mu_1 is None:
            from sklearn.ensemble import RandomForestRegressor
            self.mu_1=RandomForestRegressor(random_state=0)
        if self.mu_minus1 is None:
            from sklearn.ensemble import RandomForestRegressor
            self.mu_minus1=RandomForestRegressor(random_state=0)
        if self.param_grid_mu is None:
            self.param_grid_mu = {'min_samples_leaf': [1,3,5,10],
                                  'n_estimators': [5,10,50,100,300],
                                  'max_depth': [5,10,50,100,None]}
        
        if self.verbose:
            print("Start fitting mu_1.", flush=True)
        
        grid_1 = GridSearchCV(self.mu_1, self.param_grid_mu, cv=5, verbose=int(self.verbose), n_jobs=-1)
        grid_1.fit(self.X[self.T==1], self.Y[self.T==1])
        self.mu_1 = grid_1.best_estimator_
        
        if self.verbose:
            print("Finished fitting mu_1.", flush=True)
            print(" ", flush=True)
        
        if self.verbose:
            print("Start fitting mu_minus1.", flush=True)
        
        grid_minus1 = GridSearchCV(self.mu_minus1, self.param_grid_mu, cv=5, verbose=int(self.verbose), n_jobs=-1)
        grid_minus1.fit(self.X[self.T==-1], self.Y[self.T==-1])
        self.mu_minus1 = grid_minus1.best_estimator_

        if self.verbose:
            print("Finished fitting mu_minus1.", flush=True)
            print(" ", flush=True)

        mu_hat_1 = self.mu_1.predict(self.X).reshape(-1,1)
        mu_hat_minus1 = self.mu_minus1.predict(self.X).reshape(-1,1)
        
        mu_hat_T=np.asarray([mu_hat_1[k] if self.T[k] == 1 else mu_hat_minus1[k] for k in range(self.num_datapoints)])
        
        return mu_hat_1, mu_hat_minus1, mu_hat_T
    
    def estimate_e(self):
        e = LogisticRegression()
        e.fit(self.X, self.T)
        est_prop = e.predict_proba(self.X)
        e_hat_T = np.asarray([est_prop[k,1] if self.T[k] == 1 else est_prop[k,0] for k in range(self.num_datapoints)])
        
        return e_hat_T
    
    def predict(self,X_new):
        if self.optimal_inds is None:
            raise RuntimeError('IOPL was not fitted')
            
        num_datapoints_new=X_new.shape[0]
        pred=-np.ones(num_datapoints_new)
        for i in range(num_datapoints_new):
            for k in self.optimal_inds:
                box=self.K[k]
                helper_bool=True
                for j in range(self.d):
                    if X_new[i,j]>box[1][j] or X_new[i,j]<box[0][j]:
                        helper_bool=False
                        break
                if helper_bool:
                    pred[i]=1
        return pred
    
    def Solve_MILP(self, K, S):

        # Variables
        MILP=gp.Model("MILP",env=self.env)
        Xi=MILP.addVars(self.num_datapoints,lb=0.0,ub=1.0,vtype=GRB.CONTINUOUS)
        s=MILP.addVars(len(K),lb=0.0,ub=1.0,vtype=GRB.BINARY)
        
        # Constraints
        for i in self.I1:
            if i in self.P:
                rhs=Xi[i]
                for k in range(len(K)):
                    if S[i,k]==1.0:
                        rhs+=s[k]
                MILP.addConstr(rhs>=1.0)
        
        for i in self.I0:
            if i in self.P:
                for k in range(len(K)):
                    if S[i,k]==1.0:
                        MILP.addConstr(Xi[i]>=s[k])
                    
        for i in self.I1:
            if i in self.N:
                for k in range(len(K)):
                    if S[i,k]==1.0:
                        MILP.addConstr(1-Xi[i]-s[k]>=0)
                    
        for i in self.I0:
            if i in self.N:
                rhs=-Xi[i]
                for k in range(len(K)):
                    if S[i,k]==1.0:
                        rhs+=s[k]
                MILP.addConstr(rhs>=0.0)
        
        rhs=0
        for k in range(len(K)):
            rhs-=s[k]
        MILP.addConstr(rhs>=-self.M)
        
        # Objective
        obj=0
        for i in range(self.num_datapoints):
            obj+=Xi[i]*self.psi[i]
        MILP.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        MILP.update()
        MILP.optimize()
        
        # Get solution
        optimal_inds=[]
        for k in range(len(K)):
            val=s[k].x
            if val>0.5:
                optimal_inds.append(k)
        optimal_obj_val_MILP=MILP.objVal
        
        return optimal_inds, optimal_obj_val_MILP
    
    def fit(self,max_bnb_iterations=50):
        
        self.max_iter_for_branch_and_bound=max_bnb_iterations
        
        # Run branch and price algorithm
        data={'X': self.X,
              'Y':self.Y,
              'T': self.T,
              'M':self.M,
              'num_datapoints': self.num_datapoints,
              'd': self.d,
              'psi': self.psi,
              'I1': self.I1,
              'I0': self.I0,
              'P': self.P,
              'N': self.N,
              'epsilon': self.epsilon}
        
        initial_K=self.initial_K
        initial_S=self.initial_S
        
        initial_LP=LP(self.env, data, initial_K, initial_S, [], print_message="(LP 0) ", verbose=self.verbose)
        
        active_list=[initial_LP]
        subproblem_counter=1
        iteration_counter=0
        
        if self.verbose:
            print("Start optimization.", flush=True)
            print(" ", flush=True)
        
        while len(active_list)>0 and (iteration_counter<self.max_iter_for_branch_and_bound or self.optimal_inds is None):
            
            if self.verbose:
                print("Number of open problems: "+str(len(active_list)), flush=True)
                print(" ", flush=True)
            
            current_LP=active_list[0]
            current_LP.perform_Column_Generation(self.max_iter_for_CG)
            
            current_sol=current_LP.optimal_sol.copy()
            current_obj_val=current_LP.optimal_obj_val
            current_K=current_LP.K.copy()
            current_S=current_LP.S.copy()
            current_cutting_planes=current_LP.cutting_planes.copy()
            self.total_CG_iterations+=current_LP.total_iterations
            self.total_time_limit_exceeded_CG+=current_LP.total_time_limit_exceeded
            
            # In the first iteration solve MILP to get initial integral solution
            if iteration_counter==0:
                optimal_inds, optimal_obj_val_MILP = self.Solve_MILP(current_K.copy(),current_S.copy())
                self.K=current_K.copy()
                self.S=current_S.copy()
                self.optimal_inds=optimal_inds
                self.optimal_obj_val_MILP=optimal_obj_val_MILP
                if self.verbose:
                    print("New best objective value: "+str(self.optimal_obj_val_MILP), flush=True)
                    print(" ", flush=True)
            
            # Check if current problem can be ignored
            if current_obj_val<=self.optimal_obj_val_MILP:
            
                # Check if solution is integral
                if [round(x) for x in current_sol]==current_sol:
                    # if solution is integral update optimal_inds, optimal_obj_val_MILP, K and S
                    if self.verbose:
                        print("Solution is integral.", flush=True)
                    if current_obj_val<self.optimal_obj_val_MILP:
                        if self.verbose:
                            print("New best objective value: "+str(current_obj_val), flush=True)
                        self.optimal_obj_val_MILP=current_obj_val
                        self.optimal_inds=[ind for ind,val in enumerate(current_sol) if val>0.5]
                        self.K=current_K.copy()
                        self.S=current_S.copy()
                    if self.verbose:
                        print(" ", flush=True)
    
                else:
                    # If solution is not integral branch
                    if self.verbose:
                        print("Solution is fractional.", flush=True)
                        print(" ", flush=True)
                    
                    # Check if MILP solution with current K is better than current solution
                    optimal_inds, optimal_obj_val_MILP = self.Solve_MILP(current_K.copy(),current_S.copy())
                    if optimal_obj_val_MILP<self.optimal_obj_val_MILP:
                        if self.verbose:
                            print("New best objective value: "+str(optimal_obj_val_MILP), flush=True)
                        self.optimal_obj_val_MILP=optimal_obj_val_MILP
                        self.optimal_inds=optimal_inds
                        self.K=current_K.copy()
                        self.S=current_S.copy()
                        if self.verbose:
                            print(" ", flush=True)
                    
                    # Branch
                    volumes=[np.prod(K[1]-K[0]) for K in current_K]
                    sorted_inds=list(np.argsort(volumes))
                    i=None
                    for j in range(len(sorted_inds)-1,-1,-1): # Loop in reverse order
                        i=sorted_inds[j]
                        if round(current_sol[i])!=current_sol[i]: # if solution is fractional in index i
                            break
                    
                    # Uncomment for most infeasible branching
                    #values=[abs(x-0.5) for x in current_sol]
                    #i=values.index(min(values))
                    
                    if i is None or round(current_sol[i])==current_sol[i]:
                        raise RuntimeError('Error in branching strategy!')
                    
                    active_list.append(LP(self.env, data, current_K, current_S, current_cutting_planes+[[i,True]], print_message="(LP "+str(subproblem_counter)+") ", verbose=self.verbose))
                    active_list.append(LP(self.env, data, current_K, current_S, current_cutting_planes+[[i,False]], print_message="(LP "+str(subproblem_counter+1)+") ", verbose=self.verbose))
                    subproblem_counter+=2
                    
            else:
                
                if self.verbose:
                    print("Problem pruned.", flush=True)
                    print(" ", flush=True)
                
            iteration_counter+=1
            active_list.pop(0)
            
        if self.verbose:
            print("Optimization finished after "+str(iteration_counter)+" iteration(s).", flush=True)
            print(" ", flush=True)
            
            
            