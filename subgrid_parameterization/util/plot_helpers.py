import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from torch import device

def plot_losses(losses,labels=['Train'],cutStartPct=0.25,zoomEndEpochs=10):
    nlosses=len(losses)
    nepochs=len(losses[0])
    epochs=range(nepochs)
    plotEpochs0=slice(int(cutStartPct*nepochs),nepochs) 
    plotEpochs1=slice(int(nepochs-zoomEndEpochs-1),nepochs)
    
    fig,ax = plt.subplots(2,nlosses+1,figsize = (18, 8))
    for il in range(nlosses):
        loss=losses[il]
        ax[0,il].plot(epochs[plotEpochs0],loss[plotEpochs0],label=labels[il])
        ax[0,-1].plot(epochs,loss,label=labels[il])
        ax[1,il].plot(epochs[plotEpochs1],loss[plotEpochs1],label=labels[il])
        ax[1,-1].plot(epochs[plotEpochs1],loss[plotEpochs1],label=labels[il])
        ax[0,il].legend()
        ax[1,il].legend()
    ax[0,-1].legend()
    ax[1,-1].legend()
    
def plot_fields(dataset,model):
    ## Plot some predictions vs truth
    q,s_true=dataset.__getitem__(dataset.valid_idx[0])
    s_true=s_true.detach().squeeze().numpy()
    s_pred=model(q.unsqueeze(0)).detach().squeeze().numpy()

    fig, axs = plt.subplots(2,3,figsize=(12,6))
    ax=axs[0][0].imshow(q[0], cmap='bwr')
    fig.colorbar(ax, ax=axs[0][0])
    axs[0][0].set_xticks([]); axs[0][0].set_yticks([])
    axs[0][0].set_title("q field")

    ax=axs[0][1].imshow(s_true[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][1])
    axs[0][1].set_xticks([]); axs[0][1].set_yticks([])
    axs[0][1].set_title("True S")

    ax=axs[0][2].imshow(s_pred[0], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[0][2])
    axs[0][2].set_xticks([]); axs[0][2].set_yticks([])
    axs[0][2].set_title("Predicted S")

    ax=axs[1][0].imshow(q[1], cmap='bwr')
    fig.colorbar(ax, ax=axs[1][0])
    axs[1][0].set_xticks([]); axs[1][0].set_yticks([])

    ax=axs[1][1].imshow(s_true[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][1])
    axs[1][1].set_xticks([]); axs[1][1].set_yticks([])

    ax=axs[1][2].imshow(s_pred[1], cmap='bwr', interpolation='none')
    fig.colorbar(ax, ax=axs[1][2])
    axs[1][2].set_xticks([]); axs[1][2].set_yticks([])
    fig.tight_layout()
    return fig
    
def xp2_xpyp_term_dp1_lhs_quickPlots(model,dataset,y_text=["U2DFSN","V2DFSN"],device=device('cpu')):
    l1='Predicted'
    l2='True'

    x_test, y_test = dataset.tensors
    x_test=x_test.to(device)
    y_test=y_test.detach().cpu().numpy()
    y_pred=model(x_test).detach().cpu().numpy()

    nvar=len(y_text)
    nz=len(y_test[0])//nvar
    z=np.arange(nz)/nz
    r2=np.empty((nvar,nz))
    r=np.empty((nvar,nz))
    fig1,ax1 = plt.subplots(1,nvar,figsize = (12, 6))
    fig2,ax2 = plt.subplots(1,nvar,figsize = (12, 6))
    fig3,ax3 = plt.subplots(1,nvar,figsize = (12, 6))
    for ivar in range(nvar):
        for k in range(nz):
            r2[ivar,k]=r2_score(y_test[:,ivar*nz+k], y_pred[:,ivar*nz+k])
            r[ivar,k]=np.corrcoef(y_test[:,ivar*nz+k], y_pred[:,ivar*nz+k])[0, 1]
        
        ax1[ivar].plot(r2[ivar],z,'b',linewidth=2)
        ax1[ivar].set_title(y_text[ivar],fontsize=20)
        ax1[ivar].set_xlabel(r'R$^2$',fontsize=20)
        ax1[ivar].set_ylabel(r'$\frac{z}{z_{\text{top}}}$',rotation=0,fontsize=20)

        print("Avg. across all levels for "+y_text[ivar])
        print("R^2: %.4f" % np.mean(r2[ivar]) )
        print("Correlation: %.4f" % +np.mean(r[ivar])+"\n")
                      
        ax2[ivar].scatter(y_test[:,ivar*nz:(ivar+1)*nz], y_pred[:,ivar*nz:(ivar+1)*nz])
        xmin,xmax=ax2[ivar].get_xlim()
        ymin,ymax=ax2[ivar].get_ylim()
        ax2[ivar].plot([xmin,xmax],[xmin,xmax])
        ax2[ivar].set_xlim([xmin,xmax])
        ax2[ivar].set_xlabel(l2)
        ax2[ivar].set_ylabel(l1)
        ax2[ivar].set_title(y_text[ivar],fontsize=20)

        for it in range(y_pred.shape[0]):
            ax3[ivar].plot(y_pred[it,ivar*nz:(ivar+1)*nz],z,'r',label=l1)
            ax3[ivar].plot(y_test[it,ivar*nz:(ivar+1)*nz],z,'k',label=l2)
            l1,l2='__nolegend__','__nolegend__'
            ax3[ivar].set_xlabel(y_text[ivar],fontsize=20)
            ax3[ivar].set_ylabel(r'$\frac{z}{z_{\text{top}}}$',rotation=0,fontsize=20)
        ax3[0].legend()
        l1='Predicted'
        l2='True'

    return fig1, fig2, fig3

def UP_CBL_quickPlots(model,loader,y_text,device):
    l1='Predicted'
    l2='True'
    for data in loader:
        x_test, y_test = data
        x_test=x_test.to(device)
        y_test=y_test.detach().cpu().numpy()
        y_pred=model(x_test).detach().cpu().numpy()
        nz=int(y_pred.shape[1]/2+1)
        z=np.arange(nz-1)/nz
        fig2,ax2 = plt.subplots(1,len(y_text),figsize = (12, 6))
        fig,ax = plt.subplots(1,len(y_text),figsize = (12, 6))
        for i in range(len(y_text)):
            r2=np.mean([r2_score(y_test[:,i*(nz-1)+k], y_pred[:,i*(nz-1)+k]) for k in range(nz-1)])
            r=np.mean([np.corrcoef(y_test[:,i*(nz-1)+k], y_pred[:,i*(nz-1)+k])[0, 1] for k in range(nz-1)])
            print("Skills for "+y_text[i])
            print("R^2: %.4f" % r2 )
            print("Correlation: %.4f" % +r+"\n")

            ax2[i].scatter(y_test[:,i*nz:(i+1)*nz], y_pred[:,i*nz:(i+1)*nz])
            xmin,xmax=ax2[i].get_xlim()
            ymin,ymax=ax2[i].get_ylim()
            ax2[i].plot([xmin,xmax],[xmin,xmax])
            ax2[i].set_xlim([xmin,xmax])
            ax2[i].set_xlabel(l2)
            ax2[i].set_ylabel(l1)
            ax2[i].set_title(y_text[i],fontsize=20)
        
        for i in range(len(y_text)):                  
            for it in range(y_pred.shape[0]):
                ax[i].plot(y_pred[it,i*(nz-1):(1+i)*(nz-1)],z,'r',label=l1)
                ax[i].plot(y_test[it,i*(nz-1):(1+i)*(nz-1)],z,'k',label=l2)
                l1,l2='__nolegend__','__nolegend__'
                ax[i].set_xlabel(y_text[i],fontsize=20)
                ax[i].set_ylabel(r'$\frac{z}{z_i}$',rotation=0,fontsize=20)
            ax[0].legend()


