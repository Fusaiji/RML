% ĳһ���������ݶ�
function[f]=grad(X,Y,i,alpha,b,C,type,gamma,K,l)
k=kernel(X,X(i,:),type,gamma,1);
% e=Y(i,:)*k.*alpha;%����eӦ���Ǹ���
e=Y(i,:)*k'*alpha-1;%���޸�
% f=1/l*K*alpha+2*C*b*Y(i,:)*k.*e./((1+b*e.*e).^2);
f=1/l*K*alpha+2*C*b*Y(i,:)*e/((1+b*e*e)^2)*k;%���޸�
end
