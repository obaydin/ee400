import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import glob
import random

from mypath import Path
#from my_dataloader import mymake_data_loader
# from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
# from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.lovasz_losses import lovasz_softmax

class Trainer(object):
    def __init__(self, args, train_loader,val_loader):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        #kwargs = {'num_workers': args.workers, 'pin_memory': True}
        #self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        # =============================================================================
        #         Düzenle
        # =============================================================================
        
        

        
        
        # self.device= torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.train_loader, self.val_loader, self.test_loader, self.nclass = train_loader, val_loader, None, 2

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride)
        
        #,
                        # sync_bn=args.sync_bn,
                        # freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.AdamW(model.parameters(), args.lr)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = lovasz_softmax
        
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        args.resume=r"C:\Users\STAJYER\Desktop\models\pytorch-deeplab-xception-master\run\pascal\deeplab-resnet\experiment_10\checkpoint.pth.tar"
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            
            
    def model_forward(self,data):
        inputs=data.to(self.device)
        y_pred=self.model(inputs.float())
        return y_pred
    
    
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        running_corrects=0
        jaccard_acc_num=0
        true_pred_num=0
        data_size=0
        true_label_num=0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target=target[0]
            # image=image._squeeze(0)
            # if self.args.cuda:
            #     image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            
            
            # print(image.size())
            # output = self.model(image.float())
            output = self.model_forward(image.squeeze(0))
            # argmax_output = torch.argmax(output, dim=1)
            # uns_target=target.unsqueeze(1)
            y=target.reshape(-1).to(self.device)
            
            #accuracy hesabı
            _,preds_bnr=torch.max(output,1)
            preds_bnr_t= preds_bnr.view(-1)
            
            running_corrects += torch.sum(preds_bnr_t==y)
            jaccard_acc_num += torch.sum(torch.add(preds_bnr_t,y)==1)
            true_pred_num += torch.sum(torch.add(preds_bnr_t,y)==2)
            true_label_num+=torch.sum(y)
            
            data_size+=800*125
            
            jaccard_ind= float(true_pred_num)/float(jaccard_acc_num+true_pred_num + 0.1)
            true_acc = float(true_pred_num)/float(true_label_num+0.01)
            
            criterion = nn.CrossEntropyLoss(weight= torch.from_numpy
                                            (np.array([(300*torch.sum(target).cpu().numpy())+1,
                                                            np.product(target.size())])).float().to(self.device))
            
            #accuracy hesabı
            
            # loss = criterion(output.view(2,-1).T, y.view(-1).long())
            loss=self.criterion(output,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Jaccard Index: %.5f True_acc: %.3f  Loss: %.3f' % (jaccard_ind, true_acc,(train_loss/(i+1))) )
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 1) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', jaccard_ind, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('acc: %.5f' % jaccard_ind)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        running_corrects=0
        jaccard_acc_num=0
        true_pred_num=0
        data_size=0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            target=target[0]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image.float())
                
                
                
            y=target.reshape(-1).to(self.device)
            
            #accuracy hesabı
            _,preds_bnr=torch.max(output,1)
            preds_bnr_t= preds_bnr.view(-1)
            
            running_corrects += torch.sum(preds_bnr_t==y)
            jaccard_acc_num += torch.sum(torch.add(preds_bnr_t,y)==1)
            true_pred_num += torch.sum(torch.add(preds_bnr_t,y)==2)
            data_size+=800*125
            
            jaccard_ind= float(true_pred_num)/float(jaccard_acc_num+true_pred_num+0.1)
            
            
            loss = self.criterion(output, y)
            test_loss += loss.item()
            tbar.set_description('Test jac acc: %.5f' % (jaccard_ind ))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = jaccard_ind
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
            
            

class MyDataGenerator(Dataset):
    def __init__(self, data_file_paths, label_file_paths):
        self.data_file_paths = data_file_paths
        self.label_file_paths = label_file_paths
        
    def __len__(self):
        return len(self.data_file_paths)
    
    def __getitem__(self,idx):
        data=[]
        y=[]
        data.append(np.load(self.data_file_paths[idx]).astype(np.float32))
        data= (data-np.mean(data))/np.std(data)
        data=np.double(data)
        sample={}
        sample["image"]= data
        y.append(np.load(self.label_file_paths[idx]).astype(int))
        sample["label"]= y
        return sample





def main():
    #dosya isimlerini çekme
    #DÜZENLE
    #burada dataları çekip train valid test diye ayır
    
    
    matching_folders=[]
    
    root_directory=r"C:\Users\STAJYER\Desktop\data_staj_faster"

    
    target_name='binary'
    
    for foldername, subfolders, filenames in os.walk(root_directory):
        if target_name in subfolders:
            matching_folders.append(os.path.join(foldername,target_name))
            
    original_list=[]
    for m in range(len(matching_folders)):
        temp= glob.glob(os.path.join(matching_folders[m],'**\\*.npy'),recursive=True)
        original_list.append(temp)
        
        
    
    flattened_list=[]
    for sublist in original_list:
        for item in sublist:
            flattened_list.append(item)
    flattened_list=random.sample(flattened_list,len(flattened_list))
    
    flattened_list_specs=[]
    for l in tqdm(range(len(flattened_list))):
        bnr_direc= flattened_list[l]        
        spec_direc=bnr_direc.replace("binary", "specs")
        spec_direc= spec_direc[:-7]
        spec_direc =spec_direc +".npy"
        flattened_list_specs.append(spec_direc)
    
    val_size=int(len(flattened_list_specs)/5)
    
    val_files_bnr=flattened_list[0:val_size]
    val_files_specs=flattened_list_specs[0:val_size]
    
    train_files_bnr=flattened_list[val_size:]
    train_files_specs=flattened_list_specs[val_size:]
    
    
# =============================================================================
#     #calışıyor mu denemek için
#     train_files_bnr=train_files_bnr[0:48*10]
#     train_files_specs=train_files_specs[0:48*10]
#     
#     val_files_bnr=val_files_bnr[0:48*10]
#     val_files_specs=val_files_specs[0:48*10]
#     #calışıyor mu denemek için
# =============================================================================
    
    train_set=MyDataGenerator(train_files_specs,train_files_bnr)
    val_set=MyDataGenerator(val_files_specs,val_files_bnr)
    
    
    
    
    
    train_loader=DataLoader(train_set,batch_size=16,shuffle=False,num_workers=4)
    val_loader=DataLoader(val_set,batch_size=16,shuffle=False,num_workers=4)
    
    #DÜZENLE
    
    
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            print("using gpu")
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 1000,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.003,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args,train_loader,val_loader)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
   
   