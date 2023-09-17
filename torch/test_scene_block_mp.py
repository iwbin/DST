from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import random
import torch
import numpy as np
import gc
import time
import data_util
import scene_dataloader
import model
import loss as loss_util
from math import ceil
import pdb
from math import floor
# python test_scene.py --gpu 0 --input_data_path ./data/mp_sdf_vox_2cm_input --target_data_path ./data/mp_sdf_vox_2cm_target --test_file_list filelists/mp_rooms_1.txt --output output/mp


# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--gpu', type=int, default=4, help='which gpu to use')
parser.add_argument('--input_data_path', default="",required=False, help='path to input data')
parser.add_argument('--target_data_path', default="",required=False, help='path to target data')
parser.add_argument('--test_file_list',default='../filelists/rebuttal_scannet.txt', required=False, help='path to file list of test data')
parser.add_argument('--model_path',default="./model-epoch-4.pth", required=False, help='path to model to test')
parser.add_argument('--output', default='./stage1out', help='folder to output predictions')
#real
# model params
parser.add_argument('--num_hierarchy_levels', type=int, default=4, help='#hierarchy levels.')
parser.add_argument('--max_input_height', type=int, default=128, help='max height in voxels')
parser.add_argument('--truncation', type=float, default=3, help='truncation in voxels')
parser.add_argument('--input_dim', type=int, default=128, help='voxel dim.')
parser.add_argument('--encoder_dim', type=int, default=8, help='pointnet feature dim')
parser.add_argument('--coarse_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--refine_feat_dim', type=int, default=16, help='feature dim')
parser.add_argument('--no_pass_occ', dest='no_pass_occ', action='store_true')
parser.add_argument('--no_pass_feats', dest='no_pass_feats', action='store_true')
parser.add_argument('--use_skip_sparse', type=int, default=1, help='use skip connections between sparse convs')
parser.add_argument('--use_skip_dense', type=int, default=1, help='use skip connections between dense convs')
# test params
parser.add_argument('--compute_pred_occs', type=int, default=0, help='whether to compute occ')
parser.add_argument('--max_to_vis', type=int, default=59, help='max num to vis')
parser.add_argument('--cpu', dest='cpu', action='store_true')


parser.set_defaults(no_pass_occ=False, no_pass_feats=False, cpu=False)
args = parser.parse_args()
assert( not (args.no_pass_feats and args.no_pass_occ) )
assert( args.num_hierarchy_levels > 1 )
args.input_nf = 1
print(args)

# specify gpu
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
UP_AXIS = 0 # z is 0th


# create model
model = model.GenModel(args.encoder_dim, args.input_dim, args.input_nf, args.coarse_feat_dim, args.refine_feat_dim, args.num_hierarchy_levels, not args.no_pass_occ, not args.no_pass_feats, args.use_skip_sparse, args.use_skip_dense)
if not args.cpu:
    model = model.cuda()
checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['state_dict'])
print('loaded model:', args.model_path)
def  cut_pro_block_scence(inputs,orig_dims,loss_weights,sample):
    a1=orig_dims
    Z,Y,X=orig_dims[0]
    start=np.zeros(128*Y*X)
    start = start[:, np.newaxis]
    point_num=np.zeros(128*Y*X)
    if args.compute_pred_occs:
        start_occ=[None]*args.num_hierarchy_levels
        point_num_occ=[None]*args.num_hierarchy_levels
        for h in range(args.num_hierarchy_levels):
            factor1 = 2 ** (args.num_hierarchy_levels - h - 1)
            start_occ[h]=np.zeros((128//factor1)*(Y//factor1)*(X//factor1))
            point_num_occ[h] = np.zeros((128//factor1)*(Y//factor1)*(X//factor1))
    stride_x=56#64#56
    stride_y=56#64#56
    wide_x = 64
    wide_y = 64#64
    for y in range(int(ceil(float(Y-wide_y)/stride_y))+1):
        for x in range(int(ceil(float(X-wide_x)/stride_x))+1):
            mask_y=np.array(inputs[0][:,1]>=y*stride_y)  & np.array(inputs[0][:,1]<(y*stride_y+wide_y))
            mask_x = np.array(inputs[0][:, 2] >= x * stride_x) & np.array(inputs[0][:, 2] < (x * stride_x + wide_x))
            mask=mask_y & mask_x
            sum_point = np.sum(mask)
            print(y*(int(ceil(float(X-wide_x)/stride_x))+1)+x+1,'--',(int(ceil(float(Y-wide_y)/stride_y))+1)*(int(ceil(float(X-wide_x)/stride_x))+1))
            if(sum_point>1000):
                block_input_locs=inputs[0][mask]
                block_input_feats=inputs[1][mask]
                block_input_locs=np.array(block_input_locs[:])-[0,y*stride_y,x*stride_x,0]
                block_input_locs=torch.from_numpy(block_input_locs)
                input=[block_input_locs, block_input_feats]
                input[1] = input[1].cuda()
                output_sdf, output_occs = model(input, loss_weights)
                if args.compute_pred_occs:
                    pred_occs = [None] * args.num_hierarchy_levels
                    for h in range(args.num_hierarchy_levels):
                        if len(output_occs[h][0]) == 0:
                            continue
                        output_occs[h][1] = torch.nn.Sigmoid()(output_occs[h][1][:, 0].detach()) > 0.5
                        locs=output_occs[h][0][:,:-1]
                        vals=output_occs[h][1]
                        pred_occs[h]=locs[vals.view(-1)]
                if args.compute_pred_occs:
                    for h in range(args.num_hierarchy_levels):
                        factor=2**(args.num_hierarchy_levels-1-h)
                        pred_occs[h]=pred_occs[h][:]+torch.IntTensor([0,y*stride_y//factor,x*stride_x//factor])
                        #remove padding
                        mask_occ=(pred_occs[h][:,0]<(Z//factor))&(pred_occs[h][:,1]<(Y//factor))&(pred_occs[h][:,2]<(X//factor))
                        pred_occs[h]=pred_occs[h][mask_occ]
                        max_z_occ = Z//factor
                        max_y_occ = Y//factor
                        max_x_occ = X//factor
                        # flatten locs
                        pred_occs[h] = pred_occs[h][:, 0] * max_y_occ * max_x_occ + pred_occs[h][:, 1] * max_x_occ + pred_occs[h][:, 2]
                        # end append to start
                        locs = pred_occs[h].view(-1).cpu().numpy()
                        start_occ[h][locs] = locs
                        point_num_occ[h][locs] = 1

                output_sdf[0]=output_sdf[0][:]+torch.IntTensor([0,y*stride_y,x*stride_x,0])
                #remove padding
                mask = (output_sdf[0][:, 0] <Z) & (output_sdf[0][:, 1] < Y) & (
                            output_sdf[0][:, 2] < X)
                output_sdf[0] = output_sdf[0][mask]
                output_sdf[1] = output_sdf[1][mask]
                max_z=Z
                max_y=Y
                max_x=X
                #flatten locs
                output_sdf[0]=output_sdf[0][:,0]*max_y*max_x+output_sdf[0][:,1]*max_x+output_sdf[0][:,2]
                # end append to start
                locs= output_sdf[0].view(-1).cpu().numpy()

                point_num[locs]+=1
                end = np.zeros(128 * Y * X)
                end[locs]=output_sdf[1].view(-1).cpu().numpy()
                #point_mean.append(end)
                end = end[:, np.newaxis]
                print('ready for append')
                start=np.append(start,end,axis=1)
                start = start.sum(axis=1)[:,np.newaxis]
    start=np.squeeze(start)
    mask_loc=(point_num!=0)
    #print('divide')
    mean_feature=start[mask_loc]/point_num[mask_loc]
    #get locs
    mean_loc=np.zeros(128 * Y * X)
    mean_loc[mask_loc]=1
    mean_loc=np.nonzero(mean_loc)
    mean_loc=np.array(mean_loc,dtype=int)
    mean_loc=np.squeeze(mean_loc)
    if args.compute_pred_occs:
        mean_loc_occ=[None] * args.num_hierarchy_levels
        for h in range(args.num_hierarchy_levels):
            factor = 2 ** (args.num_hierarchy_levels - 1 - h)
            #mask_loc=(start_occ[h]!=0)
            mean_loc_occ[h]=np.nonzero(start_occ[h])
            mean_loc_occ[h] = np.array(mean_loc_occ[h], dtype=int)
            mean_loc_occ[h] = np.squeeze(mean_loc_occ[h])
            #print('mean_loc_occ',mean_loc_occ)
            z_loc_occ = mean_loc_occ[h] // ((max_y//factor) * (max_x//factor)).numpy()
            #print('z_loc_occ',z_loc_occ)
            y_loc_occ = mean_loc_occ[h] % ((max_y//factor) * (max_x//factor)).numpy() // (max_x//factor).numpy()
            x_loc_occ = mean_loc_occ[h] % (max_x//factor).numpy()
            zyx = np.array([z_loc_occ, y_loc_occ, x_loc_occ])
            zyx=zyx.transpose()
            mean_loc_occ[h]=zyx
            mean_loc_occ[h]=mean_loc_occ[h][np.newaxis,:]
            #print('mean_loc_occ[h]',mean_loc_occ[h].shape,mean_loc_occ[h])
    z_loc=mean_loc//(max_y*max_x).numpy()
    y_loc=mean_loc%(max_y*max_x).numpy()//max_x.numpy()#[:,np.newaxis]
    x_loc=mean_loc%max_x.numpy()#[:,np.newaxis]
    b_loc=np.zeros(len(x_loc),dtype=np.int)#[:,np.newaxis]
    zyxb=np.array([z_loc,y_loc,x_loc,b_loc])
    zyxb = zyxb.transpose()
    if not args.compute_pred_occs:
        mean_loc_occ=None
    return[zyxb,mean_feature],mean_loc_occ
def test(loss_weights, dataloader, output_vis, num_to_vis,start):
    model.eval()
    print('start',start)

    num_vis = 0
    num_batches = len(dataloader)
    with torch.no_grad():
        for t, sample in enumerate(dataloader):
            print('id', start+t)
            inputs = sample['input']
            if len(inputs[0])<1000:
                continue
            max1=torch.max(inputs[0][:,0])#127
            max2=torch.max(inputs[0][:,1])#782
            max3=torch.max(inputs[0][:,2])#640
            max4=torch.max(inputs[0][:,3])#0
            min1 = torch.min(inputs[0][:, 0])
            #print()
            sdf=sample['sdf']
            input_dim=np.array([128,64,64])
            sys.stdout.write('\r[ %d | %d ] %s (%d, %d, %d)    ' % (num_vis, num_to_vis, sample['name'], input_dim[0], input_dim[1], input_dim[2]))
            sys.stdout.flush()
            hierarchy_factor = pow(2, args.num_hierarchy_levels-1)
            model.update_sizes(input_dim, input_dim // hierarchy_factor)
            orig_dims = sample['orig_dims']

            try:
                inputs[1] = inputs[1].cuda()
                output_sdf,output_occ= cut_pro_block_scence(inputs, orig_dims, loss_weights, sample)
                output_sdf=[torch.from_numpy(output_sdf[0]),(torch.from_numpy(output_sdf[1])).float()]
            except:

                print('exception at %s' % sample['name'])
                gc.collect()
                continue

            # remove padding
            dims = sample['orig_dims'][0]
            orig_dims = sample['orig_dims']
            mask = (output_sdf[0][:,0] < dims[0]) & (output_sdf[0][:,1] < dims[1]) & (output_sdf[0][:,2] < dims[2])
            output_sdf[0] = output_sdf[0][mask]
            output_sdf[1] = output_sdf[1][mask]
            mask = (inputs[0][:,0] < dims[0]) & (inputs[0][:,1] < dims[1]) & (inputs[0][:,2] < dims[2])
            inputs[0] = inputs[0][mask]
            inputs[1] = inputs[1][mask]
            vis_pred_sdf = [None]
            if len(output_sdf[0]) > 0:
                vis_pred_sdf[0] = [output_sdf[0].cpu().numpy(), output_sdf[1].squeeze().cpu().numpy()]
                print('vis_pred_sdf',vis_pred_sdf)
            inputs = [inputs[0].numpy(), inputs[1].cpu().numpy()]
            data_util.save_predictions(output_vis, sample['name'], inputs, None, None, vis_pred_sdf, None, sample['world2grid'], args.truncation)
            print(start+t,'done','withoutglobal')
    sys.stdout.write('\n')


def main():
    test_files, _ = data_util.get_train_files(args.input_data_path, args.test_file_list, '')
    start=0
    test_files = test_files[start:]#200
    print('#test files = ', len(test_files))
    test_dataset = scene_dataloader.SceneDataset(test_files, args.input_dim, args.truncation, args.num_hierarchy_levels, args.max_input_height, 0, args.target_data_path)
    print('len(test_dataset)',len(test_dataset))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=scene_dataloader.collate)
    if os.path.exists(args.output):
        input('warning: output dir %s exists, press key to overwrite and continue' % args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start testing
    print('starting testing...')
    loss_weights = np.ones(args.num_hierarchy_levels+1, dtype=np.float32)
    test(loss_weights, test_dataloader, args.output, args.max_to_vis,start)



if __name__ == '__main__':
    main()


