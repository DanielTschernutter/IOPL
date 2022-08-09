import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shapely.geometry as sg
import shapely.ops as so

def T_dependent_outcome(X,T,scenario_id):
    if scenario_id==1:
        return 2.0*(X[:,1]-0.5*X[:,0]*X[:,0]+0.25)*-T
    if scenario_id==2:
        return 2.0*((4/10)**2-X[:,0]*X[:,0]-X[:,1]*X[:,1])*(X[:,0]*X[:,0]+X[:,1]*X[:,1]-(9/10)**2)*-T
    if scenario_id==3:
        return (3.0*(1-X[:,0])*(1-X[:,0])*np.exp(-X[:,0]*X[:,0]-(X[:,1]+1)*(X[:,1]+1))-10.0*(X[:,0]*0.2-X[:,0]*X[:,0]*X[:,0]-X[:,1]*X[:,1]*X[:,1]*X[:,1]*X[:,1])*np.exp(-X[:,0]*X[:,0]-X[:,1]*X[:,1])-1/3*np.exp(-(X[:,0]+1)*(X[:,0]+1)-X[:,1]*X[:,1])-1)*-T

def mean_outcome(X):
    return 1.0+2*X[:,0]+X[:,1]

def get_optimal_decision(x,scenario_id):
    x=x.reshape(-1,1).transpose()
    val1=float(mean_outcome(x)+T_dependent_outcome(x,np.array(1).reshape(-1,1),scenario_id))
    val_minus1=float(mean_outcome(x)+T_dependent_outcome(x,np.array(-1).reshape(-1,1),scenario_id))
    if val1<val_minus1:
        return 1
    else:
        return -1
    
def get_simulated_data(scenario_id,seed=0,num_of_samples=10000):
    np.random.seed(seed)
    X=np.random.uniform(-1,1,(num_of_samples,2))
    T=np.random.randint(2, size=num_of_samples)
    T[T==0]=-1
    mu_Y=mean_outcome(X)+T_dependent_outcome(X,T,scenario_id)
    Y=np.random.normal(mu_Y, 0.1, num_of_samples)
    return X,T,Y

def plot_results(IOPL_instance,scenario_id,X):
    
    pred=IOPL_instance.predict(X)
    outcome_diff=np.zeros_like(pred)
    
    for i in range(X.shape[0]):
        opt_treatment=get_optimal_decision(X[i,:],scenario_id)
        x=X[i,:].reshape(-1,1).transpose()
        outcome_diff[i]=T_dependent_outcome(x,pred[i],scenario_id)-T_dependent_outcome(x,opt_treatment,scenario_id)
    
    regret=np.mean(outcome_diff)
    
    num_datapoints_for_training=np.shape(IOPL_instance.X)[0]
    fig, ax = plt.subplots(1, 1, figsize=(8,8), constrained_layout=True)
    
    data=pd.DataFrame(columns={'X0','X1','Y','opt_treatment'})
    for i in range(num_datapoints_for_training):
        opt_treatment=get_optimal_decision(IOPL_instance.X[i,:],scenario_id)
        data=data.append({'X0': IOPL_instance.X[i,0],'X1': IOPL_instance.X[i,1],'Y': IOPL_instance.Y[i],'opt_treatment': opt_treatment}, ignore_index=True)
    
    markers = {1.0: "s", -1.0: "o"}
    sns.scatterplot(data=data,x='X0',y='X1', style='opt_treatment', markers=markers, legend=False, color=".2")
    
    rects=[]
    for ind,k in enumerate(IOPL_instance.K):
        if ind in IOPL_instance.optimal_inds:
            rects.append(sg.box(k[0][0],k[0][1],k[1][0],k[1][1]))
            
    decision_region = so.cascaded_union(rects)
    
    if decision_region.geom_type == 'MultiPolygon':
        for geom in decision_region.geoms:    
            xs, ys = geom.exterior.xy    
            ax.fill(xs, ys, alpha=0.5, fc='black', ec='none')
    else:
        xs, ys = decision_region.exterior.xy    
        ax.fill(xs, ys, alpha=0.5, fc='black', ec='none')
        
    if scenario_id==1:
        xs=np.linspace(-1,1,1000)
        ys=0.5*xs*xs-0.25
        ax.plot(xs, ys, color='black', linewidth=3)
    if scenario_id==2:
        xs=np.linspace(-1,1,100000)
        ys=np.sqrt((4/10)**2-xs*xs)
        ax.plot(xs, ys, color='black', linewidth=3)
        ys=-np.sqrt((4/10)**2-xs*xs)
        ax.plot(xs, ys, color='black', linewidth=3)
        ys=np.sqrt((9/10)**2-xs*xs)
        ax.plot(xs, ys, color='black', linewidth=3)
        ys=-np.sqrt((9/10)**2-xs*xs)
        ax.plot(xs, ys, color='black', linewidth=3)
    if scenario_id==3:
        xs=np.linspace(-1,1,100)
        ys=np.linspace(-1,1,100)
        xs,ys=np.meshgrid(xs,ys)
        zs=np.array([(3.0*(1-xs[i,j])*(1-xs[i,j])*np.exp(-xs[i,j]*xs[i,j]-(ys[i,j]+1)*(ys[i,j]+1))-10.0*(xs[i,j]*0.2-xs[i,j]*xs[i,j]*xs[i,j]-ys[i,j]*ys[i,j]*ys[i,j]*ys[i,j]*ys[i,j])*np.exp(-xs[i,j]*xs[i,j]-ys[i,j]*ys[i,j])-1/3*np.exp(-(xs[i,j]+1)*(xs[i,j]+1)-ys[i,j]*ys[i,j])-1) for i in range(100) for j in range(100)]).reshape(100,100)
        ax.contour(xs,ys,zs,[0.0],colors=['black'])
        
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$X_1$')
    plt.title('Regret: '+str(regret))