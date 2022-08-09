import gurobipy as gp
from gurobipy import GRB
import numpy as np

class LP:
    def __init__(self, env, data, K, S, cutting_planes, print_message="", verbose=True):
        
        self.X=data['X']
        self.Y=data['Y']
        self.T=data['T']
        self.M=data['M']
        self.num_datapoints=data['num_datapoints']
        self.d=data['d']
        self.psi=data['psi']
        self.I1=data['I1']
        self.I0=data['I0']
        self.P=data['P']
        self.N=data['N']
        self.epsilon=data['epsilon']
        
        self.env=env
        self.K=K.copy()
        self.S=S.copy()
        self.cutting_planes=cutting_planes.copy()
        self.print_message=print_message
        self.verbose=verbose
        
        self.optimal_sol=None
        self.optimal_obj_val=None
        self.Pricing_problem_time_limit=180
        self.total_iterations=0
        self.total_time_limit_exceeded=0
        self.r=None
        self.Pricing_inds=None
        self.Pricing_inds_pos=None
        self.Pricing_inds_neg=None
        self.Pricing_weights_pos=None
        self.Pricing_weights_neg=None
        self.Pricing_num_datapoints=None
        self.X_vals=None
        self.Kt=None
        
    def check_in_box(self,i,current_lb,current_ub):
        helper_bool=True
        for j in range(self.d):
            if self.X[i,j]>current_ub[j] or self.X[i,j]<current_lb[j]:
                helper_bool=False
                break
        return int(helper_bool)
    
    def update_r_and_inds(self,lag_mu1,lag_mu2,lag_mu3,lag_mu4):
        
        self.r=[]
        self.Pricing_inds=[]
        self.Pricing_inds_pos=[]
        self.Pricing_inds_neg=[]
        self.Pricing_weights_pos=[]
        self.Pricing_weights_neg=[]
        
        counter=-1
        for i in self.I1:
            if i in self.P:
                counter+=1
                if lag_mu1[counter]>0.0:
                    self.Pricing_inds.append(i)
                    self.Pricing_inds_pos.append(i)
                    self.Pricing_weights_pos.append(lag_mu1[counter])  
        counter=-1
        for i in self.I0:
            if i in self.P:
                counter+=1
                if lag_mu2[counter]>0.0:
                    self.Pricing_inds.append(i)
                    self.Pricing_inds_neg.append(i)
                    self.Pricing_weights_neg.append(lag_mu2[counter])              
        counter=-1
        for i in self.I1:
            if i in self.N:
                counter+=1
                if lag_mu3[counter]>0.0:
                    self.Pricing_inds.append(i)
                    self.Pricing_inds_neg.append(i)
                    self.Pricing_weights_neg.append(lag_mu3[counter])         
        counter=-1
        for i in self.I0:
            if i in self.N:
                counter+=1
                if lag_mu4[counter]>0.0:
                    self.Pricing_inds.append(i)
                    self.Pricing_inds_pos.append(i)
                    self.Pricing_weights_pos.append(lag_mu4[counter])
                    
        self.Pricing_num_datapoints=len(self.Pricing_inds)
        self.X_vals=[]
        self.Kt=[]
        for t in range(self.d):

            x_t=self.X[self.Pricing_inds,t]
            x_t_unique = np.unique(x_t)
            r_t=[int(np.where(x_t[ind]==x_t_unique)[0]) for ind in range(len(self.Pricing_inds))]
            
            self.X_vals.append(x_t)
            self.Kt.append(x_t_unique.shape[0])
            self.r.append(r_t)
    
    def Solve_Master_LP(self):

        # Variables
        MLP=gp.Model("MLP",env=self.env)
        Xi=MLP.addVars(self.num_datapoints,lb=0.0,ub=1.0,vtype=GRB.CONTINUOUS)
        s=MLP.addVars(len(self.K),lb=0.0,ub=1.0,vtype=GRB.CONTINUOUS)
        
        # Constraints
        num_constraints_for_mu1=0
        for i in self.I1:
            if i in self.P:
                rhs=Xi[i]
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        rhs+=s[k]
                MLP.addConstr(rhs>=1.0)
                num_constraints_for_mu1+=1
        
        num_constraints_for_mu2=0
        for i in self.I0:
            if i in self.P:
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        MLP.addConstr(Xi[i]>=s[k])
                        num_constraints_for_mu2+=1
                    
        num_constraints_for_mu3=0
        for i in self.I1:
            if i in self.N:
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        MLP.addConstr(1-Xi[i]-s[k]>=0)
                        num_constraints_for_mu3+=1
                    
        num_constraints_for_mu4=0
        for i in self.I0:
            if i in self.N:
                rhs=-Xi[i]
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        rhs+=s[k]
                MLP.addConstr(rhs>=0.0)
                num_constraints_for_mu4+=1
        
        rhs=0
        for k in range(len(self.K)):
            rhs-=s[k]
        MLP.addConstr(rhs>=-self.M)
        
        for cut in self.cutting_planes:
            if cut[1]:
                MLP.addConstr(s[cut[0]]>=1.0)
            else:
                MLP.addConstr(-s[cut[0]]>=0.0)
        
        # Objective
        obj=0
        for i in range(self.num_datapoints):
            obj+=Xi[i]*self.psi[i]
        MLP.setObjective(obj, GRB.MINIMIZE)
        
        # Solve
        MLP.update()
        MLP.optimize()
        
        # Get solution
        self.optimal_sol=[]
        for k in range(len(self.K)):
            val=s[k].x
            if abs(val-1)<self.epsilon:
                val=1
            if abs(val)<self.epsilon:
                val=0
            self.optimal_sol.append(val)
        
        # Get dual variables
        dual_sols=MLP.Pi
        # Numerical zeros to zero
        dual_sols=[x if abs(x)>self.epsilon else 0.0 for x in dual_sols]
        
        lag_mu1=dual_sols[:num_constraints_for_mu1]
        lag_mu2_all=dual_sols[num_constraints_for_mu1:num_constraints_for_mu1+num_constraints_for_mu2]
        lag_mu3_all=dual_sols[num_constraints_for_mu1+num_constraints_for_mu2:num_constraints_for_mu1+num_constraints_for_mu2+num_constraints_for_mu3]
        lag_mu4=dual_sols[num_constraints_for_mu1+num_constraints_for_mu2+num_constraints_for_mu3:num_constraints_for_mu1+num_constraints_for_mu2+num_constraints_for_mu3+num_constraints_for_mu4]
        lag_lambda=dual_sols[num_constraints_for_mu1+num_constraints_for_mu2+num_constraints_for_mu3+num_constraints_for_mu4]
        
        lag_mu2=[]
        counter=0
        for i in self.I0:
            if i in self.P:
                val=0
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        val+=lag_mu2_all[counter]
                        counter+=1
                lag_mu2.append(val)
        
        lag_mu3=[]
        counter=0
        for i in self.I1:
            if i in self.N:
                val=0
                for k in range(len(self.K)):
                    if self.S[i,k]==1.0:
                        val+=lag_mu3_all[counter]
                        counter+=1
                lag_mu3.append(val)
                        
        obj_val=MLP.objVal
        self.optimal_obj_val=obj_val
    
        return lag_mu1, lag_mu2, lag_mu3, lag_mu4, lag_lambda, obj_val
    
    def Pricing_Problem(self,lag_mu1,lag_mu2,lag_mu3,lag_mu4,lag_lambda):
    
        self.update_r_and_inds(lag_mu1,lag_mu2,lag_mu3,lag_mu4)
        
        Pricing_model=gp.Model("Pricing model",env=self.env)
        delta=Pricing_model.addVars(self.Pricing_num_datapoints,lb=0.0,ub=1.0,vtype=GRB.BINARY)
        var_ind=[(t,j) for t in range(self.d) for j in range(self.Kt[t])]
        gamma=Pricing_model.addVars(var_ind,lb=0.0,ub=1.0,vtype=GRB.BINARY)
        p=Pricing_model.addVars(var_ind,lb=0.0,ub=1.0,vtype=GRB.BINARY)
        q=Pricing_model.addVars(var_ind,lb=0.0,ub=1.0,vtype=GRB.BINARY)
        
        # Constraints
        for i in self.Pricing_inds_pos:
            for t in range(self.d):
                Pricing_model.addConstr(delta[self.Pricing_inds.index(i)]<=gamma[t,self.r[t][self.Pricing_inds.index(i)]])
        
        for i in self.Pricing_inds_neg:
            lhs=0
            for t in range(self.d):
                lhs+=1-gamma[t,self.r[t][self.Pricing_inds.index(i)]]
            Pricing_model.addConstr(delta[self.Pricing_inds.index(i)]+lhs>=1)
        
        for t in range(self.d):
            for j in range(self.Kt[t]):
                Pricing_model.addConstr(gamma[t,j]==p[t,j]-q[t,j])
    
        for t in range(self.d):
            for j in range(self.Kt[t]-1):
                Pricing_model.addConstr(p[t,j]<=p[t,j+1])
                Pricing_model.addConstr(q[t,j]<=q[t,j+1])
        
        # Objective
        obj=0  
        for ind,i in enumerate(self.Pricing_inds_pos):
            obj+=self.Pricing_weights_pos[ind]*delta[self.Pricing_inds.index(i)]
        for ind,i in enumerate(self.Pricing_inds_neg):
            obj-=self.Pricing_weights_neg[ind]*delta[self.Pricing_inds.index(i)]            
        Pricing_model.setObjective(obj, GRB.MAXIMIZE)
        
        # Parameter Settings
        Pricing_model.setParam(GRB.Param.Cuts, 0)
        Pricing_model.setParam(GRB.Param.ImproveStartTime, 60)
        Pricing_model.setParam(GRB.Param.TimeLimit, self.Pricing_problem_time_limit)
        
        # Timelimit check
        if Pricing_model.status==9:
            time_limit_exceeded=1
        else:
            time_limit_exceeded=0
        
        # Solve
        Pricing_model.update()
        Pricing_model.optimize()
        obj_val=-(Pricing_model.objVal-lag_lambda)
            
        # Get all solutions
        boxes=[]
        delta_vals=[]
        continue_flag=True
        num_of_sol = Pricing_model.SolCount
        for e in range(num_of_sol):
            Pricing_model.setParam(GRB.Param.SolutionNumber, e)     
            if -(Pricing_model.PoolObjVal-lag_lambda)<-self.epsilon:
                current_lb=[]
                current_ub=[]
                add_flag=True
                for j in range(self.d):
                    vals=[]
                    for i in range(self.Pricing_num_datapoints):
                        if delta[i].xn>0.5:
                            vals.append(self.X_vals[j][i])
                
                    if len(vals)==0:
                        add_flag=False
                    else:
                        current_lb.append(min(vals))
                        current_ub.append(max(vals))
                
                if add_flag:
                    
                    boxes.append([np.array(current_lb),np.array(current_ub)])
                    current_delta=[self.check_in_box(i,current_lb,current_ub) for i in range(self.num_datapoints)]
                    
                    for i in range(self.num_datapoints):
                        if i in self.Pricing_inds:
                            if round(delta[self.Pricing_inds.index(i)].xn)!=round(current_delta[i]):
                                continue_flag=False
                                if self.verbose:
                                    print("Stopped because: Delta of opt problem and computed delta not equal due to rounding errors!")
                    
                    delta_vals.append(current_delta)
                
        return obj_val, boxes, delta_vals, continue_flag, time_limit_exceeded
    
    def perform_Column_Generation(self, max_iterations):
        
        if self.verbose:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(" ")
            print('Number of cutting planes: ',len(self.cutting_planes),' (|K|=',len(self.K),')')
            print(" ")
        
        obj_val_pricing=-1
        iter_count=0
        time_limit_exceeded_counter=0
        continue_flag=True
        while obj_val_pricing<-self.epsilon and iter_count<max_iterations and continue_flag:

            [lag_mu1,lag_mu2,lag_mu3,lag_mu4,lag_lambda,obj_val_master]=self.Solve_Master_LP()
            if self.verbose:
                print(self.print_message+"CG Iteration "+str(iter_count+1)+": Master LP objective value: "+str(obj_val_master)+" (lag_lambda="+str(lag_lambda)+")", flush=True)
            
            [obj_val_pricing, boxes, delta_vals, continue_flag, time_limit_exceeded]=self.Pricing_Problem(lag_mu1,lag_mu2,lag_mu3,lag_mu4,lag_lambda)
            if self.verbose:
                print("Pricing objective value: "+str(obj_val_pricing), flush=True)
                print("Found "+str(len(boxes))+" solution(s).", flush=True)
            
            if obj_val_pricing<-self.epsilon:
                solutions_used_counter=0
                for ind,box in enumerate(boxes):
                    box_in_K_flag=any([(box[0]==k[0]).all() and (box[1]==k[1]).all() for k in self.K])
                    if not box_in_K_flag:
                        self.K.append(box)
                        new_col=np.array(delta_vals[ind]).reshape(-1,1)
                        self.S=np.hstack([self.S,new_col])
                        solutions_used_counter+=1
                    else:
                        if ind==0:
                            continue_flag=False
                            if self.verbose:
                                print("Stopped beacuse: Added already existing box due to rounding errors!")
                
                if self.verbose:
                    print("Used "+str(solutions_used_counter)+" solution(s).", flush=True)
            
            
            iter_count+=1
            time_limit_exceeded_counter+=time_limit_exceeded
            
            if self.verbose:
                print(" ")
        
        [lag_mu1,lag_mu2,lag_mu3,lag_mu4,lag_lambda,obj_val_master]=self.Solve_Master_LP()
        
        self.total_iterations=iter_count
        self.total_time_limit_exceeded=time_limit_exceeded_counter
        
        if self.verbose:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(" ")