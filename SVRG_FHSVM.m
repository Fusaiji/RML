function [alpha] = SVRG_FHSVM(X, Y , b , C,gamma,type,learning_rate,itn,itn1,tol)
[l, ~] = size(X);
X = [X ones(l,1)];
alpha = zeros(l,1);
K=kernel(X,X,type,gamma,1);
for k=1:itn
    fprintf('\n*****************************iter: %d ******************************\n', k);
    temp_ksi=1-Y.*(K*alpha);
    ksi=temp_ksi;
    ksi(ksi>0)=1;ksi(ksi<=0)=0;
    grada_Rl=1/l*K*alpha-C*b/l*(K.*Y.*ksi./((1+b*temp_ksi).^2))'*ones(l,1);
    alpha_w=alpha;
    for j=1:itn1
        fprintf('\n*****************************iter1: %d ******************************\n', j);
        i=randi([1,l]);
        fprintf('\n*****************************randi: %d ******************************\n', i);
        g=grad_FHSVM(X,Y,i,alpha_w,b,C,type,gamma,K,l)-(grad_FHSVM(X,Y,i,alpha,b,C,type,gamma,K,l)-grada_Rl);
        alpha_w=alpha_w-learning_rate*g;
    end
    alpha=alpha_w;
    stopCond = max(abs(alpha));
    if (k> 1) &&  (stopCond < tol)
        disp(' !!!stopped by termination rule!!! ');
        break;
    end

end
end 
