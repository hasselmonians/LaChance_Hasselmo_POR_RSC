# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:03:33 2024

assess symmetry across 3 domains:
    1. encoding of equally spaced head directions
    2. discrete firing locations for each equally spaced head direction
    3. firing locations distributed radially with respect to environment cneter
    
will save plots for each cell in the cell-specific folder in LaChance_Hasselmo_POR_RSC/example_cells,
as well as scores for each cell in terms of 1-fold, 2-fold, 3-fold, and 4-fold symmetry in the file symmetry_scores.csv

@author: plachanc
"""

import numpy as np
import numba as nb
import os
from scipy.stats import pearsonr, norm
from scipy.optimize import minimize
from astropy.convolution.kernels import Gaussian2DKernel
from astropy.convolution import convolve
import matplotlib.pyplot as plt
from matplotlib import colors as mplcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats.mstats import pearsonr as mapearsonr
from scipy.sparse import spdiags
from scipy.interpolate import interp1d
import csv
import copy

from utilities import collect_data


def compute_diags():
    ''' create diagonal matrices for grouped penalization -- implementation 
    modified from Hardcastle 2017 '''
    
    'diagonal matrix for computing differences between adjacent circular 1D bins'

    pos_ones = np.ones(hd_bins)
    circ1 = spdiags([-pos_ones,pos_ones],[0,1],hd_bins-1,hd_bins)
    circ_diag = circ1.T * circ1
    circ_diag=np.asarray(circ_diag.todense())
    circ_diag[0] = np.roll(circ_diag[1],-1)
    circ_diag[hd_bins-1] = np.roll(circ_diag[hd_bins-2],1)

    'also one for noncircular'

    pos_ones = np.ones(dist_bins)
    noncirc1 = spdiags([-pos_ones,pos_ones],[0,1],dist_bins-1,dist_bins)
    noncirc_diag = noncirc1.T * noncirc1
    noncirc_diag = np.asarray(noncirc_diag.todense())
        
    return circ_diag, noncirc_diag
    
@nb.jit(nopython=True)
def objective(params,Xa,dists,bearings,projection_xvals,spike_train,smoothers,smooth=True,add_hd=False):
    ''' objective function '''
    
    offset_params_x = params[:30]
    offset_params_y = params[30:60]
    fr_params = params[60:90]
    
    x_offsets = np.dot(Xa, offset_params_x)
    y_offsets = np.dot(Xa, offset_params_y)
    offsets = np.arctan2(y_offsets, x_offsets)
    
    rotated_x = dists * np.cos(bearings - offsets)

    xbins = np.digitize(rotated_x, bins=projection_xvals) - 1
    Xx = np.zeros((len(xbins),30))
    for i in range(len(xbins)):
        Xx[i][xbins[i]] = 1.
        
    u = np.dot(Xx, fr_params)
    
    if add_hd:
        hd_params = params[90:120]
        u += np.dot(Xa, hd_params)
        
    rate = np.exp(u)
    
    f = np.sum(rate - spike_train * u)
    
    if smooth:
        offset_smoother = smoothers[0]
        x_smoother = smoothers[1]

        f += 10. * np.sum(np.dot(offset_params_x.T, offset_smoother) * offset_params_x)
        f += 10. * np.sum(np.dot(offset_params_y.T, offset_smoother) * offset_params_y)
        
        vector_length = np.sqrt(np.sum(offset_params_y)**2 + np.sum(offset_params_x)**2) / len(offset_params_y)

        f += 10. * vector_length

        f += 10. * np.sum(np.dot(fr_params.T, x_smoother) * fr_params)
        
        if add_hd:
            f += 10. * np.sum(np.dot(hd_params.T, offset_smoother) * hd_params)
            
    return f


def compute_projection_maps(bearings,dists,angles,spike_train,offsets0):
    
    xmaps = []
    
    projection_xvals = np.linspace(-np.max(dists),np.max(dists),30)

    for i in np.linspace(0,360,30,endpoint=False):
        
        min_angle = i - 30
        max_angle = i + 30
        
        if min_angle < 0:
            new_min = 360 + min_angle
            good_angles = np.where((angles>new_min)|(angles<max_angle))[0]
            
        elif max_angle > 360:
            new_max = max_angle - 360
            good_angles = np.where((angles>min_angle)|(angles<new_max))[0]
            
        else:
            good_angles = np.where((angles>min_angle)&(angles<max_angle))
            
        new_train = spike_train[good_angles]

        rotated_x = dists[good_angles] * np.cos(bearings[good_angles] - offsets0[good_angles])
        
        xbins = np.digitize(rotated_x, bins=projection_xvals) - 1
        spikes = np.zeros(30)
        occ = np.zeros(30)
        
        for j in range(len(new_train)):
            spikes[xbins[j]] += new_train[j]
            occ[xbins[j]] += 1.
            
        x_map = spikes/occ
        
        xmaps.append(x_map)
        
    return xmaps
        
        
def compute_mean_angle(angles, spike_train, hd_bins=30, return_curve=False):
    
    angle_edges = np.linspace(0,360,hd_bins,endpoint=False)
    angle_midpoints = angle_edges + 360./(hd_bins*2.)
    
    angle_bins = np.digitize(angles, angle_edges) - 1
    
    spikes = np.zeros(hd_bins)
    occ = np.zeros(hd_bins)

    for i in range(len(angle_bins)):
        spikes[angle_bins[i]] += spike_train[i]
        occ[angle_bins[i]] += 1.
        
    rates = spikes/occ
    
    mean_angle = np.arctan2(np.nansum(rates * np.sin(np.deg2rad(angle_midpoints)) / np.sum(rates)), np.nansum(rates * np.cos(np.deg2rad(angle_midpoints)) / np.sum(rates))) % (2. * np.pi)
        
    if return_curve:
        return rates, mean_angle
    else:
        return mean_angle
    
    
def compute_scores(hd_curve, corr_mat, opt_result, area, csv_fname):

    params = opt_result.x

    offset_params_x = params[:30]
    offset_params_y = params[30:60]
    offset_params = np.rad2deg(np.arctan2(offset_params_y, offset_params_x))%360
    offset_params = np.unwrap(offset_params,period=360)
    detrended = offset_params - np.linspace(0,360,30,endpoint=False)
    
    radial_autocorr = []
    for i in range(len(detrended)):
        radial_autocorr.append(circ_correlation(detrended,np.roll(detrended,i))[0])
        
    autocorr_interp = interp1d(np.linspace(0,360,30,endpoint=False),radial_autocorr)
    
    quad_radial_score = np.min((autocorr_interp(90), autocorr_interp(180), autocorr_interp(270))) - np.max((autocorr_interp(45), autocorr_interp(135), autocorr_interp(225), autocorr_interp(315)))
    tri_radial_score = np.min((autocorr_interp(120), autocorr_interp(240))) - np.max((autocorr_interp(60), autocorr_interp(180), autocorr_interp(300)))
    bi_radial_score = autocorr_interp(180) - np.max((autocorr_interp(90), autocorr_interp(270)))
    uni_radial_score = 1. - autocorr_interp(180)

    
    print('4-fold radial symmetry score: %s' % str(quad_radial_score))
    print('3-fold radial symmetry score: %s' % str(tri_radial_score))
    print('2-fold radial symmetry score: %s' % str(bi_radial_score))
    print('1-fold radial symmetry score: %s' % str(uni_radial_score))
    print('------------------------')

    
    gridY, gridX = np.meshgrid(-np.arange(-60,60),np.arange(-60,60))
    dist = abs(gridX + gridY)

    masked_shifty = copy.deepcopy(corr_mat)
    masked_shifty[dist>30] = np.nan
    masked_shifty = np.ma.masked_invalid(masked_shifty)
    
    masked_stable = copy.deepcopy(corr_mat)
    masked_stable[dist>30] = np.nan
    masked_stable = np.ma.masked_invalid(masked_stable)
    
    corr_autocorr = np.zeros(len(corr_mat))

    for i in range(len(corr_mat)):
        shifted = np.roll(masked_shifty,i,axis=0)
        shifted = np.roll(shifted,i,axis=1)

        r,p = mapearsonr(shifted[dist<30].flatten(),masked_stable[dist<30].flatten())
        corr_autocorr[i] = r
    
    corr_interp = interp1d(np.linspace(0,360,120,endpoint=False),corr_autocorr)
    
    quad_corr_score = np.min((corr_interp(90), corr_interp(180), corr_interp(270))) - np.max((corr_interp(45), corr_interp(135), corr_interp(225), corr_interp(315)))
    tri_corr_score = np.min((corr_interp(120), corr_interp(240))) - np.max((corr_interp(60), corr_interp(180), corr_interp(300)))
    bi_corr_score = corr_interp(180) - np.max((corr_interp(90), corr_interp(270)))
    uni_corr_score = 1. - corr_interp(180)

    
    print('4-fold HD x loc correlation score: %s' % str(quad_corr_score))
    print('3-fold HD x loc correlation score: %s' % str(tri_corr_score))
    print('2-fold HD x loc correlation score: %s' % str(bi_corr_score))
    print('1-fold HD x loc correlation score: %s' % str(uni_corr_score))
    print('------------------------')
    
    
    hd_autocorr = np.zeros(len(hd_curve))
    for i in range(len(hd_curve)):
        r,p = pearsonr(hd_curve,np.roll(hd_curve,i))
        hd_autocorr[i] = r
    
    hd_interp = interp1d(np.linspace(0,360,60,endpoint=False),hd_autocorr)
    
    quad_hd_score = np.min((hd_interp(90), hd_interp(180), hd_interp(270))) - np.max((hd_interp(45), hd_interp(135), hd_interp(225), hd_interp(315)))
    tri_hd_score = np.min((hd_interp(120), hd_interp(240))) - np.max((hd_interp(60), hd_interp(180), hd_interp(300)))
    bi_hd_score = hd_interp(180) - np.max((hd_interp(90), hd_interp(270)))
    uni_hd_score = 1. - hd_interp(180)

    
    print('4-fold HD score: %s' % str(quad_hd_score))
    print('3-fold HD score: %s' % str(tri_hd_score))
    print('2-fold HD score: %s' % str(bi_hd_score))
    print('1-fold HD score: %s' % str(uni_hd_score))
    print('------------------------')
    
    
    quad_agg_score = quad_hd_score + quad_radial_score + quad_corr_score
    tri_agg_score = tri_hd_score + tri_radial_score + tri_corr_score
    bi_agg_score = bi_hd_score + bi_radial_score + bi_corr_score
    uni_agg_score = uni_hd_score + uni_radial_score + uni_corr_score
    
    
    print('4-fold Aggregate score: %s' % str(quad_agg_score))
    print('3-fold Aggregate score: %s' % str(tri_agg_score))
    print('2-fold Aggregate score: %s' % str(bi_agg_score))
    print('1-fold Aggregate score: %s' % str(uni_agg_score))
    print('------------------------')


    
    csv_row = [area, cell]
    csv_row += [quad_radial_score, tri_radial_score, bi_radial_score, uni_radial_score]
    csv_row += [quad_corr_score, tri_corr_score, bi_corr_score, uni_corr_score]
    csv_row += [quad_hd_score, tri_hd_score, bi_hd_score, uni_hd_score]
    csv_row += [quad_agg_score, tri_agg_score, bi_agg_score, uni_agg_score]

    with open(csv_fname,'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)
        
        
    return hd_autocorr, corr_autocorr, radial_autocorr
    
                    
def hd_by_loc_correlations(angles, center_x, center_y, spike_train, x_gr = 30, y_gr = 30):
    
    heatmaps = []
    
    for i in np.arange(0,360,3):
        
        min_angle = i - 30
        max_angle = i + 30
        
        if min_angle < 0:
            new_min = 360 + min_angle
            good_angles = np.where((angles>new_min)|(angles<max_angle))[0]
            
        elif max_angle > 360:
            new_max = max_angle - 360
            good_angles = np.where((angles>min_angle)|(angles<new_max))[0]
            
        else:
            good_angles = np.where((angles>min_angle)&(angles<max_angle))
            
        xbins = np.digitize(center_x[good_angles],bins=np.linspace(np.min(center_x),np.max(center_x)+.01,x_gr+1,endpoint=True)) - 1
        ybins = np.digitize(center_y[good_angles],bins=np.linspace(np.min(center_y),np.max(center_y)+.01,y_gr+1,endpoint=True)) - 1
    
        new_train = spike_train[good_angles]
    
        spikes = np.zeros((x_gr,y_gr))
        occ = np.zeros((x_gr,y_gr))
        
        for j in range(len(new_train)):
            spikes[xbins[j],ybins[j]] += new_train[j]
            occ[xbins[j],ybins[j]] += 1./30.
            
        heatmap = spikes/occ
        
        fr_mat = convolve(heatmap,kernel=Gaussian2DKernel(x_stddev=1.5,y_stddev=1.5))
        
        linearized = fr_mat.flatten()
        
        linearized[np.isnan(linearized)] = 0
        heatmaps.append(linearized)

    corr_mat = np.zeros((len(heatmaps),len(heatmaps)))
    for i in range(len(heatmaps)):
        for j in range(len(heatmaps)):
            r,p = pearsonr(heatmaps[i],heatmaps[j])
            corr_mat[i,j] = r
            
    return corr_mat
                    

def make_hd_curve(angles,spike_train,nbins=30):
    
    angle_bins = np.digitize(angles,bins=np.arange(0,360,int(360/nbins))) - 1
    spikes = np.zeros(nbins)
    occ = np.zeros(nbins)
    
    for i in range(len(angle_bins)):
        spikes[angle_bins[i]] += spike_train[i]
        occ[angle_bins[i]] += 1./30.
        
    hd_curve = spikes/occ
    
    return hd_curve
    

def plot_modeled_cell(angles, Xa, center_x, center_y, bearings, spike_train, opt_result, destination):
    
    params = opt_result.x
    
    offset_params_x = params[:30]
    offset_params_y = params[30:60]
    offset_params = np.arctan2(offset_params_y, offset_params_x)

    for i in range(len(offset_params)-1):
        closest_diff = (offset_params[i] - offset_params[i+1] + np.pi) % (2.*np.pi) - np.pi
        if abs(offset_params[i] - offset_params[i+1]) > (.01 + abs(closest_diff)):
            
            offset_params[i+1] = offset_params[i] - closest_diff

    fr_params = params[60:90]
    hd_params = params[90:120]
    
    offsets = np.dot(Xa, offset_params)
        
    rotated_x = dists * np.cos(bearings - offsets)

    xbins = np.digitize(rotated_x, bins=projection_xvals) - 1
    Xx = np.zeros((len(xbins),30))
    for i in range(len(xbins)):
        Xx[i][xbins[i]] = 1.
            
    u = np.dot(Xx, fr_params) + np.dot(Xa, hd_params)
    rate = np.exp(u)
    
    fr_occ = np.sum(Xx,axis=0)
    fr_params[fr_occ==0] = np.nan
    
    ani_spikes = np.random.poisson(lam=rate)
    
    spike_x = center_x[ani_spikes>0]
    spike_y = center_y[ani_spikes>0]
    spike_angles = angles[ani_spikes>0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(center_x,center_y,'gray',alpha=0.6,zorder=0)
    ax.scatter(spike_x,spike_y,c=spike_angles,cmap='hsv',zorder=1,clip_on=False)
    ax.axis('off')
    ax.axis('equal')
    
    fig.savefig(destination,dpi=300)
    
    plt.close()
    
    
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
    
    
def rayleigh_r(angles):
    ''' from Batschelet 1981 
    takes angles IN DEGREES '''
    
    mr = np.nansum(np.exp(1j*np.deg2rad(angles)))/len(angles)
    mean = np.rad2deg(np.arctan2(np.imag(mr),np.real(mr)))
    r = np.abs(mr)
    
    return r, mean
    

def circ_correlation(angles1,angles2):
    ''' from SenGupta 2001 '''
    
    r1,mean1 = rayleigh_r(angles1)
    r2,mean2 = rayleigh_r(angles2)
    
    mean1 = np.deg2rad(mean1)
    mean2 = np.deg2rad(mean2)
    
    angles1 = np.deg2rad(angles1)
    angles2 = np.deg2rad(angles2)
    
    R = np.sum(np.sin(angles1 - mean1) * np.sin(angles2 - mean2)) / np.sqrt(np.sum((np.sin(angles1 - mean1)**2)) * np.sum(np.sin(angles2 -  mean2)**2 ))

    lamb20 = (1. / len(angles1)) * np.sum(np.sin(angles1 - mean1)**2)
    lamb02 = (1. / len(angles1)) * np.sum(np.sin(angles2 - mean2)**2)
    lamb22 = (1. / len(angles1)) * np.sum((np.sin(angles1 - mean1)**2) * (np.sin(angles2 - mean2)**2))

    #calculate z score
    z = R * np.sqrt(len(angles1) * lamb20 * lamb02 / lamb22)
    
    #calculate p-value from z score
    p = 2. * (1. - norm.cdf(abs(z)))
    
    return R, z, p
    
    
def make_plots(hd_curve, hd_autocorr, corr_mat, corr_autocorr, opt_result, radial_autocorr, plot_dir):
    
    ''' HD curve '''
    
    rates = copy.deepcopy(hd_curve)
    rates = np.append(rates,rates[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rates,color='red',linewidth=3.0)
    ax.set_xlim([0,30])
    ax.set_ylim([0,1.2*np.nanmax(rates)])
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_xlabel('Head direction (deg)')
    ax.set_ylabel('Firing rate (spikes/s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_dir + '/hd_curve.png',dpi=400)
    plt.close()
    
    ''' HD curve autocorrelation '''

    acorr = copy.deepcopy(hd_autocorr)
    acorr = np.append(acorr,acorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acorr,'k-')
    ax.set_xticks([0,15,30,45,60],[0,90,180,270,360])
    ax.set_yticks([-1,-.5,0,.5,1])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(plot_dir + '/hd_autocorr.png',dpi=400)
    plt.close()
    
    
    
    ''' HD x loc correlation matrix '''
    
    ''' panel H '''
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
    cbar.set_ticks([np.round(np.nanmin(corr_mat),2),1])
    ax.axis('equal')
    ax.set_xlim([0,120])
    ax.set_ylim([120,0])
    ax.set_xlabel('HD1 (deg)')
    ax.set_ylabel('HD2 (deg)')
    fig.savefig(plot_dir + '/hd_by_loc_correlations.png',dpi=400)
    plt.close()
    
    ''' HD x loc correlation matrix autocorrelation '''

    acorr = copy.deepcopy(corr_autocorr)
    acorr = np.append(acorr,acorr[0])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acorr,'k-')
    ax.set_xticks([0,30,60,90,120],[0,90,180,270,360])
    ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.set_ylim([0,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(plot_dir + '/hd_by_loc_autocorr.png',dpi=400)
    plt.close()
    
    
    ''' radial symmetry analysis '''
    
    params = opt_result.x
    offset_params_x = params[:30]
    offset_params_y = params[30:60]
    offset_params = np.rad2deg(np.arctan2(offset_params_y, offset_params_x))%360
    offset_params = np.unwrap(offset_params,period=360)    
    detrended = offset_params - np.linspace(0,360,30,endpoint=False)
    detrended = detrended - np.mean(detrended)
    
    autocorr = []
    for i in range(len(detrended)):
        autocorr.append(circ_correlation(detrended,np.roll(detrended,i))[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(offset_params,'k')
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_ylabel('Rot offset (deg)')
    ax.set_xlabel('Head direction (deg)')
    fig.savefig(plot_dir + '/GLM_rotational_offsets.png',dpi=400)
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(autocorr)+[autocorr[0]],'k')
    ax.set_xticks([0,7.5,15,22.5,30])
    ax.set_xticklabels([0,90,180,270,360])
    ax.set_ylim([-1,1])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Offset (deg)')
    plt.tight_layout()
    fig.savefig(plot_dir + '/GLM_rotational_autocorr.png',dpi=400)
    plt.close()
    

if __name__ == '__main__':
    
    x_gr = 30
    y_gr = 30
    hd_bins = 30
    dist_bins = 30
    
    smoothers = compute_diags()

    example_cells = os.getcwd() + '/example_cells'
        
    csv_row = ['area','cell']
    csv_row += ['quad_hd_score','tri_hd_score','bi_hd_score','uni_hd_score']
    csv_row += ['quad_corr_score','tri_corr_score','bi_corr_score','uni_corr_score']
    csv_row += ['quad_radial_score','tri_radial_score','bi_radial_score','uni_radial_score']
    csv_row += ['quad_agg_score','tri_agg_score','bi_agg_score','uni_agg_score']
    
    csv_fname = example_cells + '/symmetry_scores.csv'
    
    with open(csv_fname,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_row)
    
    for area in ['POR','RSC']:
        
        areadir = example_cells + '/' + area
        
        print(area)
        
        for cell in os.listdir(areadir):
            
            celldir = areadir + '/' + cell
            
            print(cell)
            
            if not os.path.isdir(celldir):
                continue
    
            timestamps,center_x,center_y,angles = collect_data.read_video_file(celldir + '/tracking_data.txt')
            trial_data = {'timestamps':timestamps,'center_x':center_x,'center_y':center_y,'angles':angles}
            spike_timestamps = np.arange(timestamps[0],timestamps[len(timestamps)-1],1000.)
        
            trial_data['spike_timestamps'] = spike_timestamps
    
            center_x=np.array(trial_data['center_x'])
            center_y=np.array(trial_data['center_y'])
            angles=np.array(trial_data['angles'])
        
            trial_data = collect_data.ego_stuff(trial_data)
                    
            dists = np.sqrt((center_x-60.)**2 + (center_y-60.)**2)
            bearings = np.arctan2(center_y-60.,center_x-60.)
            
            center_ego_angles = np.array(trial_data['center_ego_angles'])
        
            projection_xvals = np.linspace(-np.max(dists),np.max(dists),30)
            
            angle_bins = np.digitize(angles,bins=np.arange(0,360,12)) - 1
            Xa = np.zeros((len(angles),30))
            for i in range(len(angle_bins)):
                Xa[i][angle_bins[i]] = 1.
                
            fname = celldir + '/' + 'spike_timestamps.txt'
            cluster_data={}
            cluster_data['spike_list'] = collect_data.ts_file_reader(fname)
            spike_data,cluster_data = collect_data.create_spike_lists(trial_data,cluster_data)
            spike_train = spike_data['ani_spikes']
    
            cdict = {}
    
            mean_center_bearing = compute_mean_angle(center_ego_angles, spike_train)
            
            offset_offset = (mean_center_bearing - np.pi) % (2.*np.pi)
            offset_params0 = np.linspace(0,2.*np.pi,30,endpoint=False)
            offset_params0 += offset_offset
            offsets0 = np.dot(Xa,offset_params0)
        
            proj_maps = compute_projection_maps(bearings, dists, angles, spike_train, offsets0)
                
            proj_maps = np.array(proj_maps)
            proj_maps[np.isnan(proj_maps)] = 1e-3
            proj_maps[proj_maps<1e-3] = 1e-3
            fr_params0 = np.log(np.mean(proj_maps,axis=0))
            
            bounds = [(-1,1)] * 60 + [(None,None)] * 30 + [(None,None)] * 30
            
            hd_curve, hd_mean_angle = compute_mean_angle(angles, spike_train, return_curve = True)
            hd_curve[hd_curve == 0] = 1e-3
            hd_params0 = np.log(hd_curve)
            
            offset_params_x0 = np.cos(offset_params0)
            offset_params_y0 = np.sin(offset_params0)
            
            params0 = np.concatenate((offset_params_x0,offset_params_y0,fr_params0,hd_params0))
            
            opt_result = minimize(objective,
                              x0=params0,
                              args=(Xa,dists,bearings,projection_xvals,spike_train,smoothers,
                                    True,True),
                              bounds=bounds,
                              method='Powell')
            
            plot_dir = celldir + '/plots'
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            
            plot_modeled_cell(angles, Xa, center_x, center_y, bearings, spike_train, opt_result, plot_dir + '/GLM_simulated_directional_spike_plot.png')
            
            corr_mat = hd_by_loc_correlations(angles, center_x, center_y, spike_train)
            
            hd_curve = make_hd_curve(angles, spike_train, nbins = 60)
            
            hd_autocorr, corr_autocorr, radial_autocorr = compute_scores(hd_curve, corr_mat, opt_result, area, csv_fname)

            plot_hd_map(center_x, center_y, angles, spike_train, plot_dir+'/directional_spike_plot.png')
            
            make_plots(make_hd_curve(angles,spike_train,nbins=30), hd_autocorr, corr_mat, corr_autocorr, opt_result, radial_autocorr, plot_dir)
        