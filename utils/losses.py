import torch
import torch.nn.functional as F
import torch.nn as nn


# class AdaptiveWingLoss(nn.Module):
#     def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
#         super(AdaptiveWingLoss, self).__init__()
#         self.omega = omega
#         self.theta = theta
#         self.epsilon = epsilon
#         self.alpha = alpha

#     def forward(self, pred, target,weight=None):
#         '''
#         :param pred: BxNxHxH
#         :param target: BxNxHxH
#         :return:
#         '''
#         y = target
#         y_hat = pred
#         delta_y = (y - y_hat).abs()
#         delta_y1 = torch.where(delta_y < self.theta,delta_y,torch.zeros_like(delta_y))
#         delta_y2 = torch.where(delta_y >= self.theta,delta_y,torch.zeros_like(delta_y))
#         y1 = torch.where(delta_y < self.theta,y,torch.zeros_like(y))
#         y2 = torch.where(delta_y >= self.theta,y,torch.zeros_like(y))
#         loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
#         A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
#             torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
#         C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
#         loss2 = A * delta_y2 - C
#         if weight:
#             loss1*=weight
#             loss2*=weight
#         return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)
    def forward(self, y_pred , y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
     
        return lossMat