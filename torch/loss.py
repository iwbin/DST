
import numpy as np
import torch
import torch.nn.functional as F

import sparseconvnet as scn

import data_util

UNK_THRESH = 2
#UNK_THRESH = 3

UNK_ID = -1
# z-y-x (note: returns unnormalized!)
def compute_normals_dense(sdf):
    assert(len(sdf.shape) == 5) # batch mode
    dims = sdf.shape[2:]#128 64 64
    sdfx = sdf[:,:,1:dims[0]-1,1:dims[1]-1,2:dims[2]] - sdf[:,:,1:dims[0]-1,1:dims[1]-1,0:dims[2]-2]
    sdfy = sdf[:,:,1:dims[0]-1,2:dims[1],1:dims[2]-1] - sdf[:,:,1:dims[0]-1,0:dims[1]-2,1:dims[2]-1]
    sdfz = sdf[:,:,2:dims[0],1:dims[1]-1,1:dims[2]-1] - sdf[:,:,0:dims[0]-2,1:dims[1]-1,1:dims[2]-1]
    return torch.cat([sdfx, sdfy, sdfz], 1)
def compute_normals_shape_dense(normal):#8 3 128 64 64
    assert (len(normal.shape) == 5)  # batch mode
    dims = normal.shape[2:]  # 128 64 64
    normalx2 = normal[:, :, 1:dims[0] - 1, 1:dims[1] - 1, 2:dims[2]] * normal[:, :, 1:dims[0] - 1, 1:dims[1] - 1, 0:dims[2] - 2]
    normaly2 = normal[:, :, 1:dims[0] - 1, 2:dims[1], 1:dims[2] - 1] * normal[:, :, 1:dims[0] - 1, 0:dims[1] - 2, 1:dims[2] - 1]
    normalz2 = normal[:, :, 2:dims[0], 1:dims[1] - 1, 1:dims[2] - 1] * normal[:, :, 0:dims[0] - 2, 1:dims[1] - 1, 1:dims[2] - 1]
    cosx=torch.sum(normalx2,dim=1).unsqueeze(1)
    cosy=torch.sum(normaly2,dim=1).unsqueeze(1)
    cosz=torch.sum(normalz2,dim=1).unsqueeze(1)
    #print('shapes',cosx.shape,cosy.shape,cosz.shape,torch.cat([cosx, cosy, cosz], 1).shape)
    return torch.cat([cosx, cosy, cosz], 1)
def compute_normal_loss(output_sdf, target_for_sdf, input_dim, use_loss_masking, known,loss_weights,weights):
    if len(output_sdf[0]) > 0 and loss_weights[-1] > 0:
        dims=[input_dim[0],input_dim[1],input_dim[2]]
        # print('output_sdf',output_sdf)
        batch_size=output_sdf[0][-1,-1]+1
        # print('aa',dims,batchsize)

        #compute output normal
        output_sdf_dense=torch.zeros(batch_size, 1, dims[0], dims[1], dims[2]).to(output_sdf[1].device)
        output_sdf_dense[output_sdf[0][:, -1], :, output_sdf[0][:, 0],output_sdf[0][:, 1], output_sdf[0][:, 2]] = output_sdf[1]
        normals_pred = compute_normals_dense(output_sdf_dense)
        #print('normal shape',normals_pred.shape) [8,3,126,62,62]
        normals_pred = torch.nn.functional.pad(normals_pred, (1, 1, 1, 1, 1, 1), value=-float('inf'))
        # normals_pred2=np.copy(normals_pred.detach().cpu().numpy())#.copy()
        # normals_pred2[normals_pred2== -float('inf')]=0
        # normals_pred2 = -torch.nn.functional.normalize(normals_pred2, p=2, dim=1, eps=1e-5, out=None)
        # print('dif',normals_pred,normals_pred2,normals_pred2[1,:,10:20,10:20,10:20])
        normals_pred = normals_pred[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1], output_sdf[0][:, 2]].contiguous()
        #print('shape',normals_pred.shape) [2431607, 3]
        normals_pred[normals_pred == -float('inf')] = 0
        normals_pred = -torch.nn.functional.normalize(normals_pred, p=2, dim=1, eps=1e-5, out=None)
        #print('normals_pred',normals_pred.shape)

        #compute pred normal cos3 normal -1 1
        normalscos_pred=torch.zeros(batch_size, 3, dims[0], dims[1], dims[2]).to(output_sdf[1].device)
        normalscos_pred[output_sdf[0][:, -1], :, output_sdf[0][:, 0], output_sdf[0][:, 1], output_sdf[0][:, 2]] = normals_pred
        normalscos_pred=compute_normals_shape_dense(normalscos_pred)
        normalscos_pred = torch.nn.functional.pad(normalscos_pred, (1, 1, 1, 1, 1, 1), value=-float('inf'))
        normalscos_pred = normalscos_pred[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]].contiguous()
        normalscos_pred[normalscos_pred == -float('inf')] = 0
        #print('pred-max-min', torch.max(normalscos_pred), torch.min(normalscos_pred))

        #compute target normal
        normals_target = compute_normals_dense(target_for_sdf)
        # print('normal shape',normals_pred.shape) [8,3,126,62,62]
        normals_target = torch.nn.functional.pad(normals_target, (1, 1, 1, 1, 1, 1), value=-float('inf'))
        normals_target = normals_target[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],
                       output_sdf[0][:, 2]].contiguous()
        # print('shape',normals_pred.shape) [2431607, 3]
        #mask = normals_target[:, 0] != -float('inf')
        normals_target[normals_target == -float('inf')] = 0
        normals_target = -torch.nn.functional.normalize(normals_target, p=2, dim=1, eps=1e-5, out=None)

        #compute target normal cos3
        normalscos_target = torch.zeros(batch_size, 3, dims[0], dims[1], dims[2]).to(output_sdf[1].device)
        normalscos_target[output_sdf[0][:, -1], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]] = normals_target
        normalscos_target = compute_normals_shape_dense( normalscos_target)
        normalscos_target = torch.nn.functional.pad(normalscos_target, (1, 1, 1, 1, 1, 1), value=-float('inf'))
        normalscos_target = normalscos_target[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]].contiguous()
        normalscos_target[normalscos_target == -float('inf')] = 0
        #print('target-max-min',torch.max(normalscos_target),torch.min(normalscos_target))

        #use loss masking
        if use_loss_masking:
            mask_known=known[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]]< UNK_THRESH
            mask_known=mask_known.view(-1)
            mask_unsee=target_for_sdf[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]].contiguous()!= UNK_ID
            #print(' mask_known.shape',mask_known.shape)
        #compute normal loss
        if weights is not None:
            weights_masked=weights[output_sdf[0][:, 3], :, output_sdf[0][:, 0], output_sdf[0][:, 1],output_sdf[0][:, 2]][mask_known]
            loss = torch.abs(normals_pred[mask_known] - normals_target[mask_known])
            #print('loss.shape',loss.shape)
            loss = torch.mean(loss * weights_masked)

            loss_cos = torch.abs(normalscos_pred[mask_known] - normalscos_target[mask_known])
            #print('loss_cos.shape',loss_cos.shape)
            loss_cos = torch.mean(loss_cos * weights_masked)
            #print('losses',loss,loss_cos)
            #print('normal loss',loss)
        return loss,loss_cos
    else:
        return 0,0
def compute_iou_loss(output_sdf, target_for_sdf, input_dim, use_loss_masking, known,loss_weights,weights):
    if len(output_sdf[0]) > 0 and loss_weights[-1] > 0:
        dims=[input_dim[0],input_dim[1],input_dim[2]]
        # print('output_sdf',output_sdf)
        batch_size=output_sdf[0][-1,-1]+1
        #output dense
        output_sdf_dense = torch.zeros(batch_size, 1, dims[0], dims[1], dims[2]).to(output_sdf[1].device)
        output_sdf_dense[:,:,:,:,:]=-float('inf')
        #print('output_sdf[1].shape',output_sdf[1].shape)
        output_sdf_dense[output_sdf[0][:, -1], :, output_sdf[0][:, 0], output_sdf[0][:, 1], output_sdf[0][:, 2]] = output_sdf[1]
        #print('shapes',output_sdf_dense.shape,target_for_sdf.shape,torch.max(output_sdf_dense),torch.min(output_sdf_dense),torch.max(target_for_sdf),torch.min(target_for_sdf),np.sum(target_for_sdf== -float('inf')))
        #mask
        mask_out_sdf_dense=torch.abs(output_sdf_dense)<3
        mask_out_target_dense=torch.abs(target_for_sdf)<3

        #print('shapes',mask_out_sdf_dense.shape,mask_out_target_dense.shape)
        #print('out_sdf',torch.sum(mask_out_sdf_dense.float()))
        #print('out_target',torch.sum(mask_out_target_dense.float()))
        if use_loss_masking:
            mask_known=known< UNK_THRESH
            #print('max min before mask',torch.max(target_for_sdf),torch.min(target_for_sdf))
            target_for_sdf1=target_for_sdf[mask_known]
            #print('max min after mask', torch.max(target_for_sdf1), torch.min(target_for_sdf1))
        target_float=mask_out_target_dense.float()
        target_num=torch.sum(target_float[mask_known])
        i_num=torch.sum(target_float[mask_known & mask_out_sdf_dense])
        iot=i_num/target_num
        #known mask
        pred_float=(mask_out_sdf_dense & mask_known & mask_out_target_dense).float()
        target_float=(mask_out_target_dense & mask_known).float()
        #
        B, C, Z, Y, X = pred_float.shape
        block_size=8
        pred_float=pred_float.reshape(B,C,Z//block_size,block_size,Y//block_size,block_size,X//block_size,block_size)
        target_float=target_float.reshape(B,C,Z//block_size,block_size,Y//block_size,block_size,X//block_size,block_size)
        pred_float=torch.sum(pred_float,axis=7)
        pred_float=torch.sum(pred_float,axis=5)
        pred_float=torch.sum(pred_float,axis=3)
        target_float=torch.sum(target_float,axis=7)
        target_float=torch.sum(target_float,axis=5)
        target_float=torch.sum(target_float,axis=3)
        #print('shapes',pred_float.shape,target_float.shape)
        mask_nonzero=target_float!=0
        iou_sparse=pred_float[mask_nonzero]/target_float[mask_nonzero]
        #print('maxmin1',torch.max(iou_sparse),torch.min(iou_sparse),torch.mean(iou_sparse))
        #iou2=torch.sum(pred_float[mask_nonzero])/torch.sum(target_float[mask_nonzero])
        #print('max,min',torch.max(iou_sparse),torch.min(iou_sparse),torch.mean(iou_sparse),iou_sparse.shape)
        #print('iot',iot,iou2)
        iou_sparse=1-iou_sparse
        #print('maxmin2', torch.max(iou_sparse), torch.min(iou_sparse),torch.mean(iou_sparse))
        #print('iou_sparse[0:10]',iou_sparse[0:10])
        #iou_sparse=iou_sparse*iou_sparse
        #print('iou_sparse[0:10]', iou_sparse[0:10])
        #print('maxmin3', torch.max(iou_sparse), torch.min(iou_sparse))
        #print('shape',iou_sparse.shape,torch.mean(iou_sparse))
        iou_sparse=torch.mean(iou_sparse)
        #std_iou=torch.std(iou_sparse)
        #print('std_iou',std_iou)
        #print('iot',iot)
        print('ious',1-iot,iou_sparse)
        return 1-iot,iou_sparse
    else:
        return 0,0
def compute_targets(target, hierarchy, num_hierarchy_levels, truncation, use_loss_masking, known):
    assert(len(target.shape) == 5)
    target_for_occs = [None] * num_hierarchy_levels
    target_for_hier = [None] * num_hierarchy_levels
    target_for_sdf = data_util.preprocess_sdf_pt(target, truncation)
    known_mask = None
    target_for_hier[-1] = target.clone()
    target_occ = (torch.abs(target_for_sdf) < truncation).float()
    if use_loss_masking:
        target_occ[known >= UNK_THRESH] = UNK_ID
    target_for_occs[-1] = target_occ

    factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        target_for_occs[h] = torch.nn.MaxPool3d(kernel_size=2)(target_for_occs[h+1])
        target_for_hier[h] = data_util.preprocess_sdf_pt(hierarchy[h], truncation)
        factor *= 2
    return target_for_sdf, target_for_occs, target_for_hier

# note: weight_missing_geo must be > 1
def compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation):
    # print("targrt for occs", len(target_for_occs),target_for_occs[-1].shape)
    # print("targrt for occs", torch.max(target_for_occs[-1][:, :, :, :, :]), torch.min(target_for_occs[-1][:, :, :, :, :]))
    num_hierarchy_levels = len(target_for_occs)
    weights = [None] * num_hierarchy_levels
    dims = target_for_occs[-1].shape[2:]
    #print('input_locs',input_locs.shape)
    flatlocs = input_locs[:,3]*dims[0]*dims[1]*dims[2] + input_locs[:,0]*dims[1]*dims[2] + input_locs[:,1]*dims[2] + input_locs[:,2]
    weights[-1] = torch.ones(target_for_occs[-1].shape, dtype=torch.int32).cuda()
    weights[-1].view(-1)[flatlocs] += 1
    #print(weights[-1].shape,torch.sum(weights[-1]))
    #print('target_for_occs[-1]',target_for_occs[-1])
    truncation=0.9
    weights[-1][torch.abs(target_for_occs[-1])>truncation]+=3# <= truncation] += 3
    # print('weight',torch.max(weights[-1].view(-1)[:]),torch.min(weights[-1].view(-1)[:]))
    # print('weight len',torch.sum(weights[-1].view(-1)[:]==2))
    #print('sum',torch.sum(weights[-1]<3))
    weights[-1] = (weights[-1] == 4).float() * (weight_missing_geo - 1) + 1

    #print('sum15',torch.sum(weights[-1]==1),torch.sum(weights[-1]==5))
    factor = 2
    for h in range(num_hierarchy_levels-2,-1,-1):
        weights[h] = weights[h+1][:,:,::2,::2,::2].contiguous()
        factor *= 2
    return weights


def apply_log_transform(sdf):
    sgn = torch.sign(sdf)
    out = torch.log(torch.abs(sdf) + 1)
    out = sgn * out
    return out


def compute_bce_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_loss_masking, truncation=3, batched=True):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    predvalues = sparse_pred_vals.view(-1)
    #print('dim',dims,torch.max(sparse_pred_locs[:,0]),torch.max(sparse_pred_locs[:,1]),torch.max(sparse_pred_locs[:,2]),torch.max(sparse_pred_locs[:,3]))
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = tgtvalues != UNK_ID
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
        if weight is not None:
            weight = weight[mask]
    else:
        tgtvalues[tgtvalues == UNK_ID] = 0
    if batched:
        loss = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = F.binary_cross_entropy_with_logits(predvalues, tgtvalues, weight=weight)
        else:
            raise
    return loss

def compute_iou_sparse_dense(sparse_pred_locs, dense_tgts, use_loss_masking, truncation=3, batched=True): 
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    corr = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    union = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    for b in range(dense_tgts.shape[0]):
        tgt = dense_tgts[b,0]
        if sparse_pred_locs[b] is None:
            continue
        predlocs = sparse_pred_locs[b]
        # flatten locs # TODO not sure whats the most efficient way to compute this...
        predlocs = predlocs[:,0] * dims[1] * dims[2] + predlocs[:,1] * dims[2] + predlocs[:,2]
        tgtlocs = torch.nonzero(tgt == 1)
        tgtlocs = tgtlocs[:,0] * dims[1] * dims[2] + tgtlocs[:,1] * dims[2] + tgtlocs[:,2]
        if use_loss_masking:
            tgtlocs = tgtlocs.cpu().numpy()
            # mask out from pred
            mask = torch.nonzero(tgt == UNK_ID)
            mask = mask[:,0] * dims[1] * dims[2] + mask[:,1] * dims[2] + mask[:,2]
            predlocs = predlocs.cpu().numpy()
            if mask.shape[0] > 0:
                _, mask, _ = np.intersect1d(predlocs, mask.cpu().numpy(), return_indices=True)
                predlocs = np.delete(predlocs, mask)
        else:
            predlocs = predlocs.cpu().numpy()
            tgtlocs = tgtlocs.cpu().numpy()
        if batched:
            corr += len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union += len(np.union1d(predlocs, tgtlocs))
        else:
            corr[b] = len(np.intersect1d(predlocs, tgtlocs, assume_unique=True)) 
            union[b] = len(np.union1d(predlocs, tgtlocs))
    if not batched:
        return np.divide(corr, union)
    if union > 0:
        return corr/union
    return -1

def compute_l1_predsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, weights, use_log_transform, use_loss_masking, known, batched=True, thresh=None):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)

    locs = sparse_pred_locs if thresh is None else sparse_pred_locs[sparse_pred_vals.view(-1) <= thresh]
    predvalues = sparse_pred_vals.view(-1) if thresh is None else sparse_pred_vals.view(-1)[sparse_pred_vals.view(-1) <= thresh]
    flatlocs = locs[:,3]*dims[0]*dims[1]*dims[2] + locs[:,0]*dims[1]*dims[2] + locs[:,1]*dims[2] + locs[:,2]
    tgtvalues = dense_tgts.view(-1)[flatlocs]
    weight = None if weights is None else weights.view(-1)[flatlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[flatlocs]
        predvalues = predvalues[mask]
        tgtvalues = tgtvalues[mask]
        if weight is not None:
            weight = weight[mask]
    if use_log_transform:
        predvalues = apply_log_transform(predvalues)
        tgtvalues = apply_log_transform(tgtvalues)
    if batched:
        if weight is not None:
            loss = torch.abs(predvalues - tgtvalues)
            loss = torch.mean(loss * weight)
        else:
            loss = torch.mean(torch.abs(predvalues - tgtvalues))
    else:
        if dense_tgts.shape[0] == 1:
            if weight is not None:
                loss_ = torch.abs(predvalues - tgtvalues)
                loss[0] = torch.mean(loss_ * weight).item()
            else:
                loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise
    return loss

# hierarchical loss 
def compute_loss(output_sdf, output_occs, target_for_sdf, target_for_occs, target_for_hier, loss_weights, truncation, use_log_transform=True, weight_missing_geo=1, input_locs=None, use_loss_masking=True, known=None, batched=True):
    assert(len(output_occs) == len(target_for_occs))
    batch_size = target_for_sdf.shape[0]
    loss = 0.0 if batched else np.zeros(batch_size, dtype=np.float32)
    losses = [] if batched else [[] for i in range(len(output_occs) + 1)]
    weights = [None] * len(target_for_occs)
    if weight_missing_geo > 1:
        weights = compute_weights_missing_geo(weight_missing_geo, input_locs, target_for_occs, truncation)
    for h in range(len(output_occs)):
        if len(output_occs[h][0]) == 0 or loss_weights[h] == 0:
            if batched:
                losses.append(-1)
            else:
                losses[h].extend([-1] * batch_size)
            continue
        cur_loss_occ = compute_bce_sparse_dense(output_occs[h][0], output_occs[h][1][:,0], target_for_occs[h], weights[h], use_loss_masking, batched=batched)
        cur_known = None if not use_loss_masking else (target_for_occs[h] == UNK_ID)*UNK_THRESH
        cur_loss_sdf = compute_l1_predsurf_sparse_dense(output_occs[h][0], output_occs[h][1][:,1], target_for_hier[h], weights[h], use_log_transform, use_loss_masking, cur_known, batched=batched)
        cur_loss = cur_loss_occ + cur_loss_sdf
        if batched:
            loss += loss_weights[h] * cur_loss
            losses.append(cur_loss.item())
        else:
            loss += loss_weights[h] * cur_loss
            losses[h].extend(cur_loss)
    # loss sdf
    if len(output_sdf[0]) > 0 and loss_weights[-1] > 0:
        cur_loss = compute_l1_predsurf_sparse_dense(output_sdf[0], output_sdf[1], target_for_sdf, weights[-1], use_log_transform, use_loss_masking, known, batched=batched)
        if batched:
            loss += loss_weights[-1] * cur_loss
            losses.append(cur_loss.item())
        else:
            loss += loss_weights[-1] * cur_loss
            losses[len(output_occs)].extend(cur_loss)
    else:
        if batched:
            losses.append(-1)
        else:
            losses[len(output_occs)].extend([-1] * batch_size)
    return loss, losses,weights

def compute_l1_tgtsurf_sparse_dense(sparse_pred_locs, sparse_pred_vals, dense_tgts, truncation, use_loss_masking, known, batched=True, thresh=None):
    assert(len(dense_tgts.shape) == 5 and dense_tgts.shape[1] == 1)
    batch_size = dense_tgts.shape[0]
    dims = dense_tgts.shape[2:]
    loss = 0.0 if batched else np.zeros(dense_tgts.shape[0], dtype=np.float32)
    pred_dense = torch.ones(batch_size * dims[0] * dims[1] * dims[2]).to(dense_tgts.device)
    fill_val = -truncation
    pred_dense.fill_(fill_val)
    if thresh is not None:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) <= thresh)
    else:
        tgtlocs = torch.nonzero(torch.abs(dense_tgts) < truncation)
    batchids = tgtlocs[:,0]
    tgtlocs = tgtlocs[:,0]*dims[0]*dims[1]*dims[2] + tgtlocs[:,2]*dims[1]*dims[2] + tgtlocs[:,3]*dims[2] + tgtlocs[:,4]
    tgtvalues = dense_tgts.view(-1)[tgtlocs]
    flatlocs = sparse_pred_locs[:,3]*dims[0]*dims[1]*dims[2] + sparse_pred_locs[:,0]*dims[1]*dims[2] + sparse_pred_locs[:,1]*dims[2] + sparse_pred_locs[:,2]
    pred_dense[flatlocs] = sparse_pred_vals.view(-1)
    predvalues = pred_dense[tgtlocs]
    if use_loss_masking:
        mask = known < UNK_THRESH
        mask = mask.view(-1)[tgtlocs]
        tgtvalues = tgtvalues[mask]
        predvalues = predvalues[mask]
    if batched:
        loss = torch.mean(torch.abs(predvalues - tgtvalues)).item()
    else:
        if dense_tgts.shape[0] == 1:
            loss[0] = torch.mean(torch.abs(predvalues - tgtvalues)).item()
        else:
            raise
    return loss

