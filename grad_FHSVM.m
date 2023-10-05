function[f]=grad_FHSVM(X,Y,i,alpha,b,C,type,gamma,K,l)
k=kernel(X,X(i,:),type,gamma,1);
e=1-Y(i,:)*k'*alpha;
sgn=e;sgn(sgn>0)=1;sgn(sgn<=0)=0;
f=1/l*K*alpha-sgn*C*b*k*Y(i,:)/((1+b*e)^2);
end
