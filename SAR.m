function [theta_hat, se, resid] = SAR(Y,X,W)
% This function calculate the QMLE of a linear SAR model based on normal distribution assumption. 
%Y: n-dimentional column vector
%theta=[lambda beta sigma^2]
%X should contain an intercep term

    [n K]=size(X);
    PX=X*((X'*X)\X');
    
    eig_W=eig(W);
    %negative log-likelihood function
    L=@(lambda)n*log((Y-lambda*W*Y)'*(eye(n)-PX)*(Y-lambda*W*Y))/2-sum(log(1-lambda*eig_W))+n*log(2*pi)/2; 
   
    lambda_hat=fminsearch(L, 0.4);
     %lambda_hat=fminunc(L, 0.4);
%    problem = createOptimProblem('fmincon','objective',L,'x0',0.3);
%    gs = GlobalSearch;  lambda_hat = run(gs,problem);
  
   
    beta_hat=(X'*X)\(X'*(Y-lambda_hat*W*Y));
    e_hat=Y-lambda_hat*W*Y-X*beta_hat;  
    resid=e_hat;
    sigma2_hat=e_hat'*e_hat/(n-K);
    theta_hat=[lambda_hat;beta_hat;sigma2_hat];
        
    %Hessian matrix
    G=W/(eye(n)-lambda_hat*W);
    Sigma=zeros(K+2);   % n* Eq(3.5) in Lee (2004)
    GXb=G*X*beta_hat;
    Sigma(1,1)=GXb'*GXb+sigma2_hat*trace((G+G)'*G);
    Sigma(1,2:(K+1))=GXb'*X;Sigma(2:(K+1),1)=Sigma(1,2:(K+1))';
    Sigma(2:(K+1),2:(K+1))=X'*X;
    Sigma(1,K+2)=trace(G);Sigma(K+2,1)=trace(G);
    Sigma(K+2,K+2)=n/2/sigma2_hat;
    Sigma=  Sigma/sigma2_hat/n;
    Sigma_inv=Sigma\eye(K+2);
    
    %Omega: Eq(3.6) in Lee (2004)
    mu2=mean(e_hat.^2);
    mu3=mean(e_hat.^3);
    mu4=mean(e_hat.^4);
    mu42=mu4-3*mu2^2;
    Omega=zeros(K+2);
    Omega(1,1)=2*mu3*(diag(G))'*GXb+mu42*sum((diag(G)).^2);
    Omega(1,2:(K+1))=mu3*(diag(G))'*X;Omega(2:(K+1),1)=Omega(1,2:(K+1))';
    Omega(1,K+2)=(0.5/mu2)*( mu3*sum(GXb) + mu42*trace(G) );
    Omega(K+2,1)=Omega(1,K+2);
    Omega(K+2,2:(K+1))=(0.5*mu3/mu2)*sum(X);
    Omega(2:(K+1),K+2)=Omega(K+2,2:(K+1))';
    Omega(K+2,K+2)=n*mu42/4/mu2^2;
    Omega=Omega/mu2^2/n;
    
    asym_var=(Sigma_inv+Sigma_inv*Omega*Sigma_inv)/n;
    se=sqrt(diag(asym_var));
end