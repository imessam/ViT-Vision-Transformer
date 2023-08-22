import numpy as np
import PIL
import datasets
import torchvision
import matplotlib.pyplot as plt
import os
import gc
import torch
import datetime
import copy
from copy import deepcopy
from time import sleep
from operator import itemgetter
from tqdm.auto import tqdm


def get_image_shape(image : PIL.Image):
    W, H = image.size
    C = 3 if image.mode == "RGB" else 1
    
    return (H, W, C)

def show_image(data : dict):
    
    image, label = data["image"], data["label"]
    
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
    plt.show()


def show_samples(dataset : datasets.DatasetDict):
    nrows, ncols = 3,3
    fig, axes = plt.subplots(nrows = nrows, ncols=ncols)
    for row in range(nrows):
        for col in range(ncols):

            idx = np.random.randint(low = 0, high = len(dataset))

            axes[row][col].imshow(dataset[idx]["image"])
            axes[row][col].set_title(label = dataset[idx]["label"])
            axes[row][col].axis("off")

    plt.show()



def image_to_patches(images_patches, res_patch, H, W, C):
    
    N = images_patches.shape[0]
    patch_width = res_patch
    n_rows = H // patch_width
    n_cols = W // patch_width

    cropped_img = images_patches[:,:n_rows * patch_width, :n_cols * patch_width, :]

    #
    # Into patches
    # [n_rows, n_cols, patch_width, patch_width, C]
    #
    patches = torch.empty(N, n_rows, n_cols, patch_width, patch_width, C).to(int)
    for chan in range(C):
        patches[..., chan] = (
            cropped_img[..., chan]
            .reshape(N, n_rows, patch_width, n_cols, patch_width)
            .permute(0, 1, 3, 2, 4)
        )
       
    return patches.view(N, -1, patch_width, patch_width, C)


def images_to_patches(images, res_patch):
    
    N = len(images)
    (H, W, C) = get_image_shape(images[0])
    images_patches = torch.zeros((N, H, W, 3))
    
    for i,image in enumerate(images):
        (H, W, C) = get_image_shape(image)
    
    
        if C == 1:
            image = image.convert('RGB')
            (H, W, C) = get_image_shape(image)
             
        image_tensor = torchvision.transforms.PILToTensor()(image).permute(1,2,0).to(int)

        images_patches[i] = image_tensor
    
    images_patches = image_to_patches(images_patches, res_patch = res_patch, H = H, W = W, C = C)
    return images_patches


def show_patches(patches):
    
    N,P = patches.shape[0], patches.shape[1]
       
    nrows, ncols = int(N**0.5),int(N**0.5)
    fig, axes = plt.subplots(nrows = nrows, ncols=ncols)
    for row in range(nrows):

        for col in range(ncols):

            idx = col + (row*nrows)
            
            axes[row][col].imshow(patches[idx,:,:,:])
            axes[row][col].axis("off")

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.1)
    plt.show()
    
    
    
def flatten_patches(patches):
    
    batches,N = patches.shape[:2]
    
    return patches.view(batches,N,-1)



def flat_to_patches(flat_patches, N, P, C):
    
    
    return flat_patches.view(-1,N,P,P,C)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def classificationLoss(output, target):
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    n_classes = output.shape[-1]
    output = output.reshape(-1, n_classes)
    target = target.reshape(-1).long()
    
    loss = loss_func(output, target)
    
    return loss


    
    
def train_model(model, trainLoaders, lossFn, optimizer,
                scheduler = None, num_epochs=1, device = "cpu", 
                isSave = False, filename = "vit-weights", verbose = True):
    
    since = datetime.datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    prev_train_loss = 0
    prev_val_loss = 0

    train_losses = []
    val_losses = []


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i,data in enumerate(tqdm(trainLoaders[phase],"Predicting ...")):

                if device != "cpu":
                    torch.cuda.empty_cache()
                
                inputs = data["image_patches_flatten"].squeeze(1).to(device)
                labels = data["label"].to(device)
                                    

                # zero the parameter gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):

                    # forward
                    outputs =  model(inputs)            
                    
                    loss = lossFn(outputs, labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        
                        # learning rate scheduler
                        if scheduler is not None:
                            scheduler.step()

                # statistics
                running_loss += (loss.item()*len(inputs))
                if verbose:
                    print(f' Iteration Loss : {loss.item()*len(inputs)}, lr = {optimizer.param_groups[0]["lr"]}')
            
            epoch_loss = running_loss / len(trainLoaders[phase])
            
            if phase == "train":
                            
                print(f"{phase} prev epoch Loss: {prev_train_loss}")
                prev_train_loss = epoch_loss
                train_losses.append(epoch_loss)
                   
            if phase == "val":
                
                print(f"{phase} prev epoch Loss: {prev_val_loss}")
                prev_val_loss = epoch_loss
                val_losses.append(epoch_loss)
                
                # deep copy the model
                if epoch_loss<best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if isSave:
                        torch.save(model.state_dict(), f"trained/{filename}")    
                        
            print(f"{phase} current epoch Loss: {epoch_loss}, lr = {optimizer.param_groups[0]['lr']}")


    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


    ## load best model weights
    # model.load_state_dict(best_model_wts)

      
    return model, train_losses, val_losses
    

    
    
def evaluate_model(model, test_data, device = "cpu"):
    since = datetime.datetime.now()
    
    model.eval()   # Set model to evaluate mode

    results = []
    # Iterate over data.
    for i,data in enumerate(tqdm(test_data,"Predicting ...")):

        inputs = data["image_patches_flatten"].squeeze(1).to(device)
        labels = data["label"].to(device)

        # forward
        outputs =  model(inputs)
    
        result = score(outputs, labels, kind = "accuracy")
        results.append(result)

    results = torch.tensor(results)

    print(f" Result : {results.mean()}")

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Evaluating complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return results


def score(output, target, kind = "accuracy"):
    
    n_classes = output.shape[-1]
    output = output.reshape(-1, n_classes)
    target = target.reshape(-1).long()
    
    output_preds = torch.argmax(torch.softmax(output,dim = -1),dim = 1)
    tp, tn, fp, fn, acc, f1 = 0, 0, 0, 0, 0, 0
    # for i,pred in enumerate(output_preds):
    #     if output_preds[i] == target[i]:
    #         tp += 1
    
    tp = (output_preds == target).sum()
    
    acc = tp/len(target)
    
    if kind == "accuracy":
        return acc
    
    