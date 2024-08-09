# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:04:28 2024

run this file to generate data-based plots from all main figures

plots are saved in a folder for the associated figure wihin LaChance_Hasselmo_Nat_Comm_2024/figures

datapoints for population analyses are in LaChance_Hasselmo_Nat_Comm_2024/figures/datapoints
data for individual example cells shown in the main figures are in LaChance_Hasselmo_Nat_Comm_2024/figures/example_cells

@author: plachanc
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pickle

framerate = 30.

sns.set_style("white")
rc('font',**{'family':'sans-serif','sans-serif':['Arial'],'size':21})
rc('xtick',**{'bottom':True,'major.size':6,'minor.size':6,'major.width':1.5,'minor.width':1.5})
rc('ytick',**{'left':True,'major.size':6,'minor.size':6,'major.width':1.5,'minor.width':1.5})
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
    

def plot_hd_map(center_x,center_y,angles,spike_train,destination):
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
    markersize = plt.rcParams['lines.markersize'] ** 2
    
    spike_x = center_x[spike_train>0]
    spike_y = center_y[spike_train>0]
    spike_angles = angles[spike_train>0]
    
    colormap = plt.get_cmap('hsv')
    norm = mplcolors.Normalize(vmin=0, vmax=360)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(center_x,center_y,color='gray',alpha=0.6,zorder=0)
    ax.scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
    ax.axis('off')
    ax.axis('equal')
    
    fig.savefig(destination,dpi=300)
    
    plt.close()
    
    
def plot_path_spikes(center_x,center_y,angles,spike_train,destination):
    
    savedir = os.path.dirname(destination)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        
    markersize = plt.rcParams['lines.markersize'] ** 2
    
    spike_x = center_x[spike_train>0]
    spike_y = center_y[spike_train>0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(center_x,center_y,color='gray',alpha=0.6,zorder=0)
    ax.scatter(spike_x,spike_y,s=markersize,c='red',zorder=1,clip_on=False)
    ax.axis('off')
    ax.axis('equal')
    
    fig.savefig(destination,dpi=300)
    
    plt.close()
    
    
def plot_stacked_hd_curves(data,target_fname,just_two=False):
    
    pre_rates = data['1.2m s1']['hd_curve']
    pre_rates = np.append(pre_rates,pre_rates[0])
    exp_rates = data['1.2m ab s2']['hd_curve']
    exp_rates = np.append(exp_rates,exp_rates[0])
    if not just_two:
        post_rates = data['1.2m s3']['hd_curve']
        post_rates = np.append(post_rates,post_rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pre_rates,color='darkred',linewidth=3.0)
    ax.plot(exp_rates,color='red',linestyle='--',linewidth=3.0,zorder=100)
    if not just_two:
        ax.plot(post_rates,color='red',linestyle='-',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylim([0,None])
    
    yticks = ax.get_yticks()
    for j in yticks:
        print(np.float(j))
        if np.float(j) == np.float(7.5):
            ax.set_yticks([0,4,8])
    
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
def plot_waveform(waveform, target_fname):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(waveform,'k-')
    ax.set_ylabel('Voltage (V)')
    ax.set_xticks([0,31],[0,1])
    ax.set_xlabel('Time (ms)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()


def make_all_figures():
    
    ''' figure 1 '''
    
    figdir = os.getcwd() + '/figures'
    
    for i in range(1,7):
        if not os.path.exists(figdir+'/fig%i' % i):
            os.mkdir(figdir+'/fig%i' % i)
    
    ''' panels F-I '''
    
    data_fname = figdir + '/example_cells/1F-I.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    ''' panel F left '''
    target_fname = figdir + '/fig1/panel_F_left.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel F right '''
    target_fname = figdir + '/fig1/panel_F_right.png'
    rates = data['ego_curve']
    rates = np.append(rates,rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rates,color='blue',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Egocentric bearing (deg)')
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,21])
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel G '''
    target_fname = figdir + '/fig1/panel_G.png'
    rates = data['hd_curve']
    rates = np.append(rates,rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rates,color='red',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,21])
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel G inset '''
    target_fname = figdir + '/fig1/panel_G_inset.png'
    autocorr = data['hd_acorr']
    autocorr = np.append(autocorr,autocorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(autocorr,'k-')
    ax.set_xticks([0,15,30,45,60],[0,90,180,270,360])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel H '''
    target_fname = figdir + '/fig1/panel_H.png'
    corr_mat = data['hd_loc_corr']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr_mat,cmap='viridis')
    ax.set_yticks([0,30,60,90,120])
    ax.set_yticklabels([0,90,180,270,360])
    ax.set_xticks([0,30,60,90,120])
    ax.set_xticklabels([0,90,180,270,360])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-.57,1])
    cbar.set_label('Correlation',rotation=270)
    # ax.axis('off')
    ax.axis('equal')
    ax.set_xlim([0,120])
    ax.set_ylim([120,0])
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel I left '''
    target_fname = figdir + '/fig1/panel_I_left.png'
    dist_params = data['dist_params']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dist_params,'k')
    ax.set_yticks([0,5,10,15,20])
    ax.set_xticks([.5,7.75,15,22.25,28.5])
    ax.set_xticklabels([-80,-40,0,40,80])
    ax.set_ylim([0,1.2*np.nanmax(dist_params)])
    ax.set_xlabel('Dist from center (cm)')
    ax.set_ylabel('Firing rate (Hz)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel I right '''
    target_fname = figdir + '/fig1/panel_I_right.png'
    rot_params = data['offset_params']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rot_params,'k')
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Rotational offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel I right inset '''
    target_fname = figdir + '/fig1/panel_I_right_inset.png'
    autocorr = data['offset_acorr']
    autocorr = np.append(autocorr,autocorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(autocorr,'k-')
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    ''' -------------------------------------------------------- '''
    

    ''' panels J-M'''
    
    data_fname = figdir + '/example_cells/1J-M.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    ''' panel J left '''
    target_fname = figdir + '/fig1/panel_J_left.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel J right '''
    target_fname = figdir + '/fig1/panel_J_right.png'
    rates = data['ego_curve']
    rates = np.append(rates,rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rates,color='blue',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Egocentric bearing (deg)')
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,16])
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel K '''
    target_fname = figdir + '/fig1/panel_K.png'
    rates = data['hd_curve']
    rates = np.append(rates,rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rates,color='red',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,16])
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel K inset '''
    target_fname = figdir + '/fig1/panel_K_inset.png'
    autocorr = data['hd_acorr']
    autocorr = np.append(autocorr,autocorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(autocorr,'k-')
    ax.set_xticks([0,15,30,45,60],[0,90,180,270,360])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel L '''
    target_fname = figdir + '/fig1/panel_L.png'
    corr_mat = data['hd_loc_corr']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr_mat,cmap='viridis')
    ax.set_yticks([0,30,60,90,120])
    ax.set_yticklabels([0,90,180,270,360])
    ax.set_xticks([0,30,60,90,120])
    ax.set_xticklabels([0,90,180,270,360])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-.51,1])
    cbar.set_label('Correlation',rotation=270)
    # ax.axis('off')
    ax.axis('equal')
    ax.set_xlim([0,120])
    ax.set_ylim([120,0])
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel M left '''
    target_fname = figdir + '/fig1/panel_M_left.png'
    dist_params = data['dist_params']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dist_params,'k')
    ax.set_yticks([0,5,10,15,20])
    ax.set_xticks([.5,7.75,15,22.25,28.5])
    ax.set_xticklabels([-80,-40,0,40,80])
    ax.set_ylim([0,1.2*np.nanmax(dist_params)])
    ax.set_xlabel('Dist from center (cm)')
    ax.set_ylabel('Firing rate (Hz)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel M right '''
    target_fname = figdir + '/fig1/panel_M_right.png'
    rot_params = data['offset_params']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rot_params,'k')
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Rotational offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel M right inset '''
    target_fname = figdir + '/fig1/panel_M_right_inset.png'
    autocorr = data['offset_acorr']
    autocorr = np.append(autocorr,autocorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(autocorr,'k-')
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()

    

    ''' figure 2 '''

    ''' panels A, C, and E '''
    
    data_fname = figdir + '/datapoints/2ACE.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    ''' panel A top left '''
    target_fname = figdir + '/fig2/panel_A_top_left.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data['POR']['sorted_curves'],cmap='viridis')
    ax.set_yticks([0,84],[1,85])
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_ylabel('Cell number')
    ax.set_aspect(.2)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel A right '''
    target_fname = figdir + '/fig2/panel_A_right.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data['RSC']['sorted_curves'],cmap='viridis')
    ax.set_yticks([0,209],[1,210])
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_xlabel('Shifted HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(.2)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel A bottom left '''
    target_fname = figdir + '/fig2/panel_A_bottom_left.png'
    por_curves = data['POR']['sorted_curves']
    rsc_curves = data['RSC']['sorted_curves']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.nanmean(por_curves,axis=0),color='purple',linewidth=3)
    ax.plot(np.nanmean(rsc_curves,axis=0),color='cornflowerblue',linewidth=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0,1])
    ax.set_yticks([0,1])
    ax.set_ylabel('Normed rate')
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.legend(['POR','RSC'])
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()


    ''' panel C left '''
    target_fname = figdir + '/fig2/panel_C_left.png'
    corr_mat = data['POR']['mean_corr_mat']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr_mat,cmap='viridis')
    ax.set_yticks([0,30,60,90,120])
    ax.set_yticklabels([0,90,180,270,360])
    ax.set_xticks([0,30,60,90,120])
    ax.set_xticklabels([0,90,180,270,360])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([np.nanmin(corr_mat),1])
    cbar.ax.set_yticklabels(['Min',1])
    cbar.set_label('Correlation',rotation=270)
    # ax.axis('off')
    ax.axis('equal')
    ax.set_xlim([0,120])
    ax.set_ylim([120,0])
    ax.set_xlabel('HD 1 (deg)')
    ax.set_ylabel('HD 2 (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel C right '''
    target_fname = figdir + '/fig2/panel_C_right.png'
    corr_mat = data['RSC']['mean_corr_mat']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(corr_mat,cmap='viridis')
    ax.set_yticks([0,30,60,90,120])
    ax.set_yticklabels([0,90,180,270,360])
    ax.set_xticks([0,30,60,90,120])
    ax.set_xticklabels([0,90,180,270,360])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([np.nanmin(corr_mat),1])
    cbar.ax.set_yticklabels(['Min',1])
    cbar.set_label('Correlation',rotation=270)
    # ax.axis('off')
    ax.axis('equal')
    ax.set_xlim([0,120])
    ax.set_ylim([120,0])
    ax.set_xlabel('HD 1 (deg)')
    ax.set_ylabel('HD 2 (deg)')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()


    ''' panel E top left '''
    target_fname = figdir + '/fig2/panel_E_top_left.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data['POR']['sorted_offsets'],cmap='viridis')
    ax.set_yticks([0,84],[1,85])
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_ylabel('Cell number')
    ax.set_aspect(.2)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel E right '''
    target_fname = figdir + '/fig2/panel_E_right.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data['RSC']['sorted_offsets'],cmap='viridis')
    ax.set_yticks([0,209],[1,210])
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.set_xlabel('Shifted HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(.2)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed offset',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel E bottom left '''
    target_fname = figdir + '/fig2/panel_E_bottom_left.png'
    por_curves = data['POR']['sorted_offsets']
    rsc_curves = data['RSC']['sorted_offsets']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.nanmean(por_curves,axis=0),color='purple',linewidth=3)
    ax.plot(np.nanmean(rsc_curves,axis=0),color='cornflowerblue',linewidth=3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([.35,.6])
    ax.set_yticks([.4,.5,.6])
    ax.set_ylabel('Normed offset')
    ax.set_xticks([0,7.5,15,22.5,30],[0,90,180,270,360])
    ax.legend(['POR','RSC'])
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()


    ''' panels B, D, F, G, H '''
    
    data_fname = figdir + '/datapoints/2BDFGH.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    ''' panels B, D, F, G '''
    
    for score_type in ['agg','hd','corr','sym']:
        
        score_data = data[score_type]
        
        if score_type == 'agg':
            panel = 'G'
            ylim = [-3,4.1]
        elif score_type == 'hd':
            panel = 'B'
            ylim = [-2,2]
        elif score_type == 'corr':
            panel = 'D'
            ylim = [-1,1]
        elif score_type == 'sym':
            panel = 'F'
            ylim = [-2,2]
            
        target_fname = figdir + '/fig2/panel_%s.png' % panel
            
        por_data = score_data['POR']
        por_len = len(por_data['quad'])
        rsc_data = score_data['RSC']
        rsc_len = len(rsc_data['quad'])

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),por_data['uni'],color='purple',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(1+np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),por_data['bi'],color='purple',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(2+np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),por_data['tri'],color='purple',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(3+np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),por_data['quad'],color='purple',linestyle='None',marker='o',markersize=4,clip_on=False)
    
        plt.plot([-.5,.5],[por_data['uni_95'],por_data['uni_95']],'r-')
        plt.plot([.5,1.5],[por_data['bi_95'],por_data['bi_95']],'r-')
        plt.plot([1.5,2.5],[por_data['tri_95'],por_data['tri_95']],'r-')
        plt.plot([2.5,3.5],[por_data['quad_95'],por_data['quad_95']],'r-')
    
        ax.plot(4+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),rsc_data['uni'],color='cornflowerblue',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(5+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),rsc_data['bi'],color='cornflowerblue',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(6+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),rsc_data['tri'],color='cornflowerblue',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(7+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),rsc_data['quad'],color='cornflowerblue',linestyle='None',marker='o',markersize=4,clip_on=False)
       
        plt.plot([3.5,4.5],[rsc_data['uni_95'],rsc_data['uni_95']],'r-')
        plt.plot([4.5,5.5],[rsc_data['bi_95'],rsc_data['bi_95']],'r-')
        plt.plot([5.5,6.5],[rsc_data['tri_95'],rsc_data['tri_95']],'r-')
        plt.plot([6.5,7.5],[rsc_data['quad_95'],rsc_data['quad_95']],'r-')
       
        ax.set_xticks([0,1,2,3,4,5,6,7])
        ax.set_ylabel('Symmetry score')
        ax.set_xticklabels(['1-fold','2-fold','3-fold','4-fold','1-fold','2-fold','3-fold','4-fold'],rotation=40)
        
        ax.plot([-.5,7.5],[0,0],'k-',zorder=0)
        ax.set_xlim([-.5,7.5])
        ax.set_ylim(ylim)
        plt.tight_layout()
        fig.savefig(target_fname, dpi=400)
        plt.close()


    ''' panel H '''
    
    target_fname = figdir + '/fig2/panel_H.png'

    rsc_counts, xbins = np.histogram(data['agg']['RSC']['quad'],bins=np.linspace(-2.5,0,15))
    por_counts, xbins = np.histogram(data['agg']['POR']['quad'],bins=np.linspace(-2.5,0,15))

    rsc_counts = rsc_counts.astype(float)/np.sum(rsc_counts.astype(float))
    por_counts = por_counts.astype(float)/np.sum(por_counts.astype(float))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(xbins[:-1],xbins,weights=rsc_counts,color='cornflowerblue')
    ax.hist(xbins[:-1],xbins,weights=por_counts,color='purple',alpha=.6)
    ax.set_ylabel('Fraction of cells')
    ax.set_xlabel('4-fold aggregate score')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    
    ''' figure 3 '''
    
    ''' panel 3C leftmost '''
    
    data_fname = figdir + '/example_cells/3C1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_C_leftmost.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 3C mid-left '''
    
    data_fname = figdir + '/example_cells/3C2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_C_mid-left.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 3C mid-right '''
    
    data_fname = figdir + '/example_cells/3C3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_C_mid-right.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 3C rightmost '''
    
    data_fname = figdir + '/example_cells/3C4.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_C_rightmost.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    
    ''' panel 3D left '''
    
    data_fname = figdir + '/datapoints/3D.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    target_fname = figdir + '/fig3/panel_D_left.png'
    por_len = len(data['POR_square'])
    rsc_len = len(data['RSC_square'])
        
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),-data['POR_square'],color='purple',markerfacecolor='None',linestyle='None',marker='o',markeredgewidth=2,clip_on=False)
    ax.plot(1+np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),-data['POR_lshape'],color='purple',linestyle='None',marker='o',clip_on=False)
    ax.plot(2+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),-data['RSC_square'],color='cornflowerblue',markerfacecolor='None',linestyle='None',marker='o',markeredgewidth=2,clip_on=False)
    ax.plot(3+np.zeros(rsc_len)+.5*(np.random.rand(rsc_len)-.5),-data['RSC_lshape'],color='cornflowerblue',linestyle='None',marker='o',clip_on=False)

    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['POR sq','POR L','RSC sq','RSC L'],rotation=30)
    
    ax.plot([-.5,3.5],[0,0],'k-',zorder=0)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([-.55,.3])
    ax.set_yticks([-.4,-.2,0,.2])

    ax.set_ylabel('Globality index')
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    
    ''' panel 3D right '''
    
    target_fname = figdir + '/fig3/panel_D_right.png'

    por_delta = -np.array(data['POR_lshape'] - data['POR_square'])
    rsc_delta = -np.array(data['RSC_lshape'] - data['RSC_square'])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.zeros(len(por_delta))+.5*(np.random.rand(len(por_delta))-.5),por_delta,color='purple',linestyle='None',marker='o',clip_on=False)
    ax.plot(1+np.zeros(len(rsc_delta))+.5*(np.random.rand(len(rsc_delta))-.5),rsc_delta,color='cornflowerblue',linestyle='None',marker='o',clip_on=False)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['POR','RSC'])
    
    ax.plot([-.5,1.5],[0,0],'k-',zorder=0)
    ax.set_xlim([-.5,1.5])
    ax.set_ylim([-.5,.23])
    ax.set_yticks([-.4,-.2,0,.2])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    ax.set_ylabel('$\Delta$ Globality index')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    
    ''' panel 3E left '''
    
    data_fname = figdir + '/example_cells/3E1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_E_left.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 3E right '''
    
    data_fname = figdir + '/example_cells/3E2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig3/panel_E_right.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    

    ''' Panel 3F-I'''
    
    data_fname = figdir + '/datapoints/3F-I.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    ''' panel 3F left '''
    
    heatmap = data['mean_heatmaps']['combined']['POR']
    
    target_fname = figdir + '/fig3/panel_F_left.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.12,vmax=.12,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.12,.12])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()

    ''' panel 3F middle '''
    
    heatmap = data['mean_heatmaps']['combined']['RSC']
    
    target_fname = figdir + '/fig3/panel_F_middle.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.12,vmax=.12,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.12,.12])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()

    ''' panel 3F right '''
    
    heatmap = data['mean_heatmaps']['combined']['MEC']
    
    target_fname = figdir + '/fig3/panel_F_right.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.12,vmax=.12,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.12,.12])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 3H leftmost '''
    
    heatmap = data['mean_heatmaps']['square']['POR']
    
    target_fname = figdir + '/fig3/panel_H_leftmost.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.14,vmax=.14,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.14,.14])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 3H mid-left '''
    
    heatmap = data['mean_heatmaps']['l_shape']['POR']
    
    target_fname = figdir + '/fig3/panel_H_mid-left.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.14,vmax=.14,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.14,.14])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 3H mid-right '''
    
    heatmap = data['mean_heatmaps']['square']['RSC']
    
    target_fname = figdir + '/fig3/panel_H_mid-right.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.14,vmax=.14,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.14,.14])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 3H rightmost '''
    
    heatmap = data['mean_heatmaps']['l_shape']['RSC']
    
    target_fname = figdir + '/fig3/panel_H_rightmost.png'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heatmap,vmin=-.14,vmax=.14,cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('left', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cax.yaxis.set_ticks_position('left')
    cbar.set_ticks([-.14,.14])
    ax.axis('off')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 3G, I '''
    
    ''' panel 3G '''
    
    target_fname = figdir + '/fig3/panel_G.png'
    
    por_near = np.array(data['near_changes']['combined']['POR'])
    por_away = np.array(data['away_changes']['combined']['POR'])
    rsc_near = np.array(data['near_changes']['combined']['RSC'])
    rsc_away = np.array(data['away_changes']['combined']['RSC'])
    mec_near = np.array(data['near_changes']['combined']['MEC'])
    mec_away = np.array(data['away_changes']['combined']['MEC'])

    por_jitter = np.random.normal(0,.07,len(por_near))
    rsc_jitter = np.random.normal(0,.07,len(rsc_near))
    mec_jitter = np.random.normal(0,.07,len(mec_near))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.zeros_like(por_jitter)+por_jitter,por_near,color='purple',marker='o',linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+1+rsc_jitter,rsc_near,color='cornflowerblue',marker='o',linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot(np.zeros_like(mec_jitter)+2+mec_jitter,mec_near,color='green',marker='o',linestyle='None',alpha=.9,zorder=102,clip_on=False)
    ax.plot(np.zeros_like(por_jitter)+3+por_jitter,por_away,color='purple',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+4+rsc_jitter,rsc_away,color='cornflowerblue',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot(np.zeros_like(mec_jitter)+5+mec_jitter,mec_away,color='green',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=102,clip_on=False)
    ax.plot([-.3,.3],[np.mean(por_near),np.mean(por_near)],'k-',linewidth=3)
    ax.plot([.7,1.3],[np.mean(rsc_near),np.mean(rsc_near)],'k-',linewidth=3)
    ax.plot([1.7,2.3],[np.mean(mec_near),np.mean(mec_near)],'k-',linewidth=3)
    ax.plot([2.7,3.3],[np.mean(por_away),np.mean(por_away)],'k-',linewidth=3)
    ax.plot([3.7,4.3],[np.mean(rsc_away),np.mean(rsc_away)],'k-',linewidth=3)
    ax.plot([4.7,5.3],[np.mean(mec_away),np.mean(mec_away)],'k-',linewidth=3)
    ax.plot([-.5,5.5],[0,0],'k-',alpha=.6,zorder=200)
    ax.set_xlim([-.5,5.5])
    ax.set_ylim([-.45,.45])
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_yticks([-.4,-.2,0,.2,.4])
    ax.set_yticklabels([-.4,-.2,0,.2,.4],size=14)
    ax.set_xticklabels(['POR','RSC','MEC/PaS','POR','RSC','MEC/PaS'],rotation=60,size=14)
    ax.set_ylabel('$\Delta$ Firing rate (normed)',size=14)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 3I left '''
    
    target_fname = figdir + '/fig3/panel_I_left.png'
    
    por_near = np.array(data['near_changes']['square']['POR'])
    por_away = np.array(data['away_changes']['square']['POR'])
    rsc_near = np.array(data['near_changes']['square']['RSC'])
    rsc_away = np.array(data['away_changes']['square']['RSC'])

    por_jitter = np.random.normal(0,.07,len(por_near))
    rsc_jitter = np.random.normal(0,.07,len(rsc_near))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.zeros_like(por_jitter)+por_jitter,por_near,color='purple',marker='o',linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+1+rsc_jitter,rsc_near,color='cornflowerblue',marker='o',linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot(np.zeros_like(por_jitter)+2+por_jitter,por_away,color='purple',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+3+rsc_jitter,rsc_away,color='cornflowerblue',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot([-.3,.3],[np.mean(por_near),np.mean(por_near)],'k-',linewidth=3)
    ax.plot([.7,1.3],[np.mean(rsc_near),np.mean(rsc_near)],'k-',linewidth=3)
    ax.plot([1.7,2.3],[np.mean(por_away),np.mean(por_away)],'k-',linewidth=3)
    ax.plot([2.7,3.3],[np.mean(rsc_away),np.mean(rsc_away)],'k-',linewidth=3)
    ax.plot([-.5,3.5],[0,0],'k-',alpha=.6,zorder=200)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([-.45,.45])
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([-.4,-.2,0,.2,.4])
    ax.set_ylabel('$\Delta$ Firing rate (normed)')
    ax.set_xticklabels(['POR','RSC','POR','RSC'],rotation=35)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    ''' panel 3I right '''
    
    target_fname = figdir + '/fig3/panel_I_right.png'
    
    por_near = np.array(data['near_changes']['l_shape']['POR'])
    por_away = np.array(data['away_changes']['l_shape']['POR'])
    rsc_near = np.array(data['near_changes']['l_shape']['RSC'])
    rsc_away = np.array(data['away_changes']['l_shape']['RSC'])

    por_jitter = np.random.normal(0,.07,len(por_near))
    rsc_jitter = np.random.normal(0,.07,len(rsc_near))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.zeros_like(por_jitter)+por_jitter,por_near,color='purple',marker='o',linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+1+rsc_jitter,rsc_near,color='cornflowerblue',marker='o',linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot(np.zeros_like(por_jitter)+2+por_jitter,por_away,color='purple',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=100,clip_on=False)
    ax.plot(np.zeros_like(rsc_jitter)+3+rsc_jitter,rsc_away,color='cornflowerblue',marker='o',markerfacecolor='None',markeredgewidth=2,linestyle='None',alpha=.9,zorder=101,clip_on=False)
    ax.plot([-.3,.3],[np.mean(por_near),np.mean(por_near)],'k-',linewidth=3)
    ax.plot([.7,1.3],[np.mean(rsc_near),np.mean(rsc_near)],'k-',linewidth=3)
    ax.plot([1.7,2.3],[np.mean(por_away),np.mean(por_away)],'k-',linewidth=3)
    ax.plot([2.7,3.3],[np.mean(rsc_away),np.mean(rsc_away)],'k-',linewidth=3)
    ax.plot([-.5,3.5],[0,0],'k-',alpha=.6,zorder=200)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([-.45,.45])
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([-.4,-.2,0,.2,.4])
    ax.set_ylabel('$\Delta$ Firing rate (normed)')
    ax.set_xticklabels(['POR','RSC','POR','RSC'],rotation=35)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()


    ''' figure 4 '''
    
    ''' panel 4A left '''
    
    data_fname = figdir + '/example_cells/4A1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig4/panel_A_left.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 4A middle '''
    
    data_fname = figdir + '/example_cells/4A2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig4/panel_A_middle.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 4A right '''
    
    data_fname = figdir + '/example_cells/4A3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)

    target_fname = figdir + '/fig4/panel_A_right.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)


    ''' panel 4C, F, I '''

    data_fname = figdir + '/datapoints/4CFI.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    ''' panel 4C left '''
    
    target_fname = figdir + '/fig4/panel_C_left.png'
    rates = np.array(data['POR']['pre'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,33],[1,34])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.4)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4C middle '''
    
    target_fname = figdir + '/fig4/panel_C_middle.png'
    rates = np.array(data['POR']['exp'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,33],[1,34])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.4)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4C right '''
    
    target_fname = figdir + '/fig4/panel_C_right.png'
    rates = np.array(data['POR']['post'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,33],[1,34])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.4)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    

    ''' panel 4F left '''
    
    target_fname = figdir + '/fig4/panel_F_left.png'
    rates = np.array(data['RSC']['pre'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,36],[1,37])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.3)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4F middle '''
    
    target_fname = figdir + '/fig4/panel_F_middle.png'
    rates = np.array(data['RSC']['exp'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,36],[1,37])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.3)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4F right '''
    
    target_fname = figdir + '/fig4/panel_F_right.png'
    rates = np.array(data['RSC']['post'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,36],[1,37])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.3)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    

    ''' panel 4I left '''
    
    target_fname = figdir + '/fig4/panel_I_left.png'
    rates = np.array(data['MEC']['pre'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,29],[1,30])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.55)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4I middle '''
    
    target_fname = figdir + '/fig4/panel_I_middle.png'
    rates = np.array(data['MEC']['exp'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,29],[1,30])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.55)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    ''' panel 4I right '''
    
    target_fname = figdir + '/fig4/panel_I_right.png'
    rates = np.array(data['MEC']['post'])
    rates = np.concatenate((rates,rates[:,0].reshape((len(rates),1))),axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(rates,cmap='viridis')
    ax.set_yticks([0,29],[1,30])
    ax.set_xticks([0,15,30],[0,180,360])
    ax.set_xlabel('HD (deg)')
    ax.set_ylabel('Cell number')
    ax.set_aspect(1.55)
    cbar = fig.colorbar(im, orientation='vertical',fraction=0.046, pad=0.04)
    cbar.set_ticks([0,1])
    cbar.set_label('Normed rate',rotation=270)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    ''' panel 4D, G, J '''
    
    data_fname = figdir + '/datapoints/4DGJ.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    ''' panel 4D '''
    
    target_fname = figdir + '/fig4/panel_D.png'
    
    exp = data['POR']['exp']
    ctl = data['POR']['ctl']
    
    data_dict = {}
    data_dict['val'] = np.concatenate((exp,ctl))
    data_dict['label'] = ['AB-A1']*len(exp) + ['A2-A1']*len(ctl)
    data_df = pd.DataFrame(data_dict)

    fig = plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['black','cornflowerblue'],clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_xlim([-.5,len(np.unique(data_df['label']))-.5])
    ax.set_ylim([-1,1.5])
    ax.set_yticks([-1,-.5,0,.5,1,1.5])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('')
    median_width = 0.5
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()  # "X" or "Y"

        # calculate the mean value for all replicates of either X or Y
        median_val = data_df[data_df['label']==var].val.mean()

        # plot horizontal lines across the column, centered on the tick
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=0,alpha=0.8)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    

    ''' panel 4G '''
    
    target_fname = figdir + '/fig4/panel_G.png'
    
    exp = data['RSC']['exp']
    ctl = data['RSC']['ctl']
    
    data_dict = {}
    data_dict['val'] = np.concatenate((exp,ctl))
    data_dict['label'] = ['AB-A1']*len(exp) + ['A2-A1']*len(ctl)
    data_df = pd.DataFrame(data_dict)

    fig = plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['black','cornflowerblue'],clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_xlim([-.5,len(np.unique(data_df['label']))-.5])
    ax.set_ylim([-1,1.5])
    ax.set_yticks([-1,-.5,0,.5,1,1.5])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('')
    median_width = 0.5
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()  # "X" or "Y"

        # calculate the mean value for all replicates of either X or Y
        median_val = data_df[data_df['label']==var].val.mean()

        # plot horizontal lines across the column, centered on the tick
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=0,alpha=0.8)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    

    ''' panel 4J '''
    
    target_fname = figdir + '/fig4/panel_J.png'
    
    exp = data['MEC']['exp']
    ctl = data['MEC']['ctl']
    
    data_dict = {}
    data_dict['val'] = np.concatenate((exp,ctl))
    data_dict['label'] = ['AB-A1']*len(exp) + ['A2-A1']*len(ctl)
    data_df = pd.DataFrame(data_dict)

    fig = plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['black','cornflowerblue'],clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_xlim([-.5,len(np.unique(data_df['label']))-.5])
    ax.set_ylim([-1,1.5])
    ax.set_yticks([-1,-.5,0,.5,1,1.5])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('')
    median_width = 0.5
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()  # "X" or "Y"

        # calculate the mean value for all replicates of either X or Y
        median_val = data_df[data_df['label']==var].val.mean()

        # plot horizontal lines across the column, centered on the tick
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=0,alpha=0.8)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    ''' panels 4E, H, K '''

    data_fname = figdir + '/datapoints/4EHK.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    ''' panel 4E '''
    
    target_fname = figdir + '/fig4/panel_E.png'
    
    cue_A_mods = data['POR']['og']
    cue_B_mods = data['POR']['new']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(cue_A_mods,cue_B_mods,zorder=1000,c='k',clip_on=False)
    ax.plot((0,1),(0,1),'k-')
    ax.set_ylabel('Cue B Mod Index')
    ax.set_xlabel('Cue A Mod Index')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    

    ''' panel 4H '''
    
    target_fname = figdir + '/fig4/panel_H.png'
    
    cue_A_mods = data['RSC']['og']
    cue_B_mods = data['RSC']['new']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(cue_A_mods,cue_B_mods,zorder=1000,c='k',clip_on=False)
    ax.plot((0,1),(0,1),'k-')
    ax.set_ylabel('Cue B Mod Index')
    ax.set_xlabel('Cue A Mod Index')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()


    ''' panel 4K '''
    
    target_fname = figdir + '/fig4/panel_K.png'
    
    cue_A_mods = data['MEC']['og']
    cue_B_mods = data['MEC']['new']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(cue_A_mods,cue_B_mods,zorder=1000,c='k',clip_on=False)
    ax.plot((0,1),(0,1),'k-')
    ax.set_ylabel('Cue B Mod Index')
    ax.set_xlabel('Cue A Mod Index')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xticks([0,.5,1])
    ax.set_yticks([0,.5,1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(target_fname,dpi=400)
    plt.close()
    
    
    ''' figure 5 '''
    
    ''' panel 5A leftmost '''
    
    data_fname = figdir + '/example_cells/5A1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_A_leftmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_A_leftmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)
    
    ''' panel 5A mid-left '''
    
    data_fname = figdir + '/example_cells/5A2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_A_mid-left.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_A_mid-left_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)

    
    ''' panel 5A mid-right '''
    
    data_fname = figdir + '/example_cells/5A3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_A_mid-right.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_A_mid-right_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)


    ''' panel 5A rightmost '''
    
    data_fname = figdir + '/example_cells/5A4.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_A_rightmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_A_rightmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)

    
    ''' panel 5B leftmost '''
    
    data_fname = figdir + '/example_cells/5B1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_B_leftmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_B_leftmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)
    
    ''' panel 5B mid-left '''
    
    data_fname = figdir + '/example_cells/5B2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_B_mid-left.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_B_mid-left_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)

    
    ''' panel 5B mid-right '''
    
    data_fname = figdir + '/example_cells/5B3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_B_mid-right.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_B_mid-right_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)


    ''' panel 5B rightmost '''
    
    data_fname = figdir + '/example_cells/5B4.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_B_rightmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_B_rightmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)
    
    
    ''' panel 5C leftmost '''
    
    data_fname = figdir + '/example_cells/5C1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_C_leftmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_C_leftmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)
    
    ''' panel 5C mid-left '''
    
    data_fname = figdir + '/example_cells/5C2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_C_mid-left.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_C_mid-left_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)

    
    ''' panel 5C mid-right '''
    
    data_fname = figdir + '/example_cells/5C3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_C_mid-right.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_C_mid-right_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)


    ''' panel 5C rightmost '''
    
    data_fname = figdir + '/example_cells/5C4.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_C_rightmost.png'
    plot_stacked_hd_curves(data,target_fname)
    
    target_fname = figdir + '/fig5/panel_C_rightmost_inset.png'
    plot_waveform(data['1.2m s1']['waveform'],target_fname)
    
    
    ''' panel 5D '''
    
    data_fname = figdir + '/datapoints/5D.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    ''' panel 5D left '''

    target_fname = figdir + '/fig5/panel_D_left.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data['POR']['latencies'],data['POR']['delt_bi'],c='purple',zorder=100)
    ax.plot([0,.6],[0,0],'k-',alpha=.7)
    ax.set_xlim([0,.6])
    ax.set_ylim([-.6,1.5])
    ax.plot([.2,.2],[-.6,1.5],'red',alpha=.7,zorder=-1)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([-.5,0,.5,1,1.5])
    ax.set_xticks([0,.2,.4,.6])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('Peak-trough latency (ms)')
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 5D middle '''
    
    target_fname = figdir + '/fig5/panel_D_middle.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data['RSC']['latencies'],data['RSC']['delt_bi'],c='cornflowerblue',zorder=100)
    ax.plot([0,.6],[0,0],'k-',alpha=.7)
    ax.set_xlim([0,.6])
    ax.set_ylim([-.6,1.5])
    ax.plot([.2,.2],[-.6,1.5],'red',alpha=.7,zorder=-1)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([-.5,0,.5,1,1.5])
    ax.set_xticks([0,.2,.4,.6])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('Peak-trough latency (ms)')
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 5D right '''
    
    target_fname = figdir + '/fig5/panel_D_right.png'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data['MEC']['latencies'],data['MEC']['delt_bi'],c='green',zorder=100)
    ax.plot([0,.6],[0,0],'k-',alpha=.7)
    ax.set_xlim([0,.6])
    ax.set_ylim([-.6,1.5])
    ax.plot([.2,.2],[-.6,1.5],'red',alpha=.7,zorder=-1)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yticks([-.5,0,.5,1,1.5])
    ax.set_xticks([0,.2,.4,.6])
    ax.set_ylabel('$\Delta$ Bidirectinality Index')
    ax.set_xlabel('Peak-trough latency (ms)')
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 5E top left '''
    
    data_fname = figdir + '/example_cells/5Etop1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_topleft.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)

    ''' panel 5E top middle '''

    data_fname = figdir + '/example_cells/5Etop2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_topmiddle.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 5E top right '''

    data_fname = figdir + '/example_cells/5Etop3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_topright.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 5E bottom left '''

    data_fname = figdir + '/example_cells/5Ebottom1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_bottomleft.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 5E bottom middle '''

    data_fname = figdir + '/example_cells/5Ebottom2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_bottommiddle.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 5E bottom right '''

    data_fname = figdir + '/example_cells/5Ebottom3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig5/panel_E_bottomright.png'
    plot_path_spikes(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    
    ''' Figure 6 '''
    
    '''panel 6A top '''
        
    data_fname = figdir + '/example_cells/6Atop.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_A_top.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    '''panel 6A bottom '''
        
    data_fname = figdir + '/example_cells/6Abottom.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_A_bottom.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)

    ''' panel 6C '''
    data_fname = figdir + '/datapoints/6C.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    for score_type in ['hd','corr','sym']:
        
        score_data = data[score_type]
        
        if score_type == 'hd':
            panel = 'left'
            ylim = [-2,2]
        elif score_type == 'corr':
            panel = 'middle'
            ylim = [-1,1]
        elif score_type == 'sym':
            panel = 'right'
            ylim = [-2,2]
            
        target_fname = figdir + '/fig6/panel_C_%s.png' % panel
            
        por_data = score_data['POR']
        por_len = len(por_data['quad'])
        rsc_data = score_data['RSC']
        rsc_len = len(rsc_data['quad'])

        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.plot(np.zeros(por_len)+.5*(np.random.rand(por_len)-.5),por_data['quad'],color='purple',linestyle='None',marker='o',markersize=4,clip_on=False)
        ax.plot(1+np.zeros(rsc_len)+.4*(np.random.rand(rsc_len)-.5),rsc_data['quad'],color='cornflowerblue',linestyle='None',marker='o',markersize=4,clip_on=False)
        plt.plot([-.3,.3],[por_data['quad_95'],por_data['quad_95']],'r-')
        plt.plot([.7,1.3],[rsc_data['quad_95'],rsc_data['quad_95']],'r-')
        ax.set_xticks([0,1],['POR','RSC'])
        ax.plot([-.5,1.5],[0,0],'k-',zorder=0)
        ax.set_xlim([-.5,1.5])
        ax.set_ylim(ylim)
        ax.set_ylabel('4-fold symmetry score')
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(1.3*(x1-x0)/(y1-y0))
        plt.tight_layout()
        fig.savefig(target_fname, dpi=400)
        plt.close()

    ''' panel 6E left '''

    data_fname = figdir + '/example_cells/6E1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_E_left.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 6E right '''

    data_fname = figdir + '/example_cells/6E2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_E_right.png'
    plot_hd_map(data['center_x'],data['center_y'],data['angles'],data['spike_train'],target_fname)
    
    ''' panel 6F '''
    
    data_fname = figdir + '/datapoints/6F.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    target_fname = figdir + '/fig6/panel_F.png'

    por_delta = -np.array(data['POR_lshape'] - data['POR_square'])
    rsc_delta = -np.array(data['RSC_lshape'] - data['RSC_square'])

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(np.zeros(len(por_delta))+.5*(np.random.rand(len(por_delta))-.5),por_delta,color='purple',linestyle='None',marker='o',clip_on=False)
    ax.plot(1+np.zeros(len(rsc_delta))+.5*(np.random.rand(len(rsc_delta))-.5),rsc_delta,color='cornflowerblue',linestyle='None',marker='o',clip_on=False)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['POR','RSC'])
    ax.plot([-.5,1.5],[0,0],'k-',zorder=0)
    ax.set_xlim([-.5,1.5])
    ax.set_ylim([-.5,.23])
    ax.set_yticks([-.4,-.2,0,.2])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    ax.set_ylabel('$\Delta$ Globality index')
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    ''' panel 6H top left '''
    
    data_fname = figdir + '/example_cells/6Htop1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_topleft.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    ''' panel 6H bottom left '''
    
    data_fname = figdir + '/example_cells/6Hbottom1.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_bottomleft.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    ''' panel 6H top middle '''
    
    data_fname = figdir + '/example_cells/6Htop2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_topmiddle.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    ''' panel 6H bottom middle'''
    
    data_fname = figdir + '/example_cells/6Hbottom2.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_bottommiddle.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    ''' panel 6H top right '''
    
    data_fname = figdir + '/example_cells/6Htop3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_topright.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    ''' panel 6H bottom right '''
    
    data_fname = figdir + '/example_cells/6Hbottom3.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
    
    target_fname = figdir + '/fig6/panel_H_bottomright.png'
    plot_stacked_hd_curves(data,target_fname,just_two=True)
    
    
    ''' panel 6I '''
    
    data_fname = figdir + '/datapoints/6I.pickle'
    with open(data_fname,'rb') as f:
        data = pickle.load(f)
        
    target_fname = figdir + '/fig6/panel_I.png'

    por_delt_bi = np.array(data['POR']['exp'])
    rsc_delt_bi = np.array(data['RSC']['exp'])
    mec_delt_bi = np.array(data['MEC']['exp'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(np.zeros(len(por_delt_bi))+.4*(np.random.rand(len(por_delt_bi))-.5),por_delt_bi,color='purple',linestyle='None',marker='o',markersize=5,clip_on=False)
    ax.plot(1+np.zeros(len(rsc_delt_bi))+.4*(np.random.rand(len(rsc_delt_bi))-.5),rsc_delt_bi,color='cornflowerblue',linestyle='None',marker='o',markersize=5,clip_on=False)
    ax.plot(2+np.zeros(len(mec_delt_bi))+.4*(np.random.rand(len(mec_delt_bi))-.5),mec_delt_bi,color='green',linestyle='None',marker='o',markersize=5,clip_on=False)
    ax.set_xticks([0,1,2])
    ax.plot([-.5,2.5],[0,0],'k-',zorder=0)
    ax.set_xlim([-.5,2.5])
    ax.set_ylim([-.5,1.5])
    ax.set_xticks([0,1,2],['POR','RSC','MEC'])
    ax.set_ylabel('$\Delta$ Bidirectionality index')
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    plt.tight_layout()
    fig.savefig(target_fname, dpi=400)
    plt.close()
    
    
            
if __name__ == '__main__':

    make_all_figures()