close all
clear all
addpath('PROPACK')
rng(0)
n = 2^16 - 1;             % dimension, n = n1+n2-1 s.t. n1=n2
r = 10;                   % rank 10
sample_rate = 0.1;        % 10% sample rate
m = ceil(sample_rate*n);  % number of samples
alpha = 0.1;              % 10% outlier
k = round(alpha*m);       % number of outliers
c = 5;                    % outlier magnitude
condition_number = 100;   % condition number 100

freq_seed = randperm(n+3, r)/(n+3); % For off-grid
sigma_star = linspace(1, 1/condition_number, r);
ox = exp(2 * pi * 1i * (0:(n-1))' * freq_seed) * sigma_star'/n;
M = randperm(n, m);

obs = ox(M);
mean_of_ox = mean(abs(ox));

K = randsample(m,k);  % outlier location
obs(K) = obs(K) + c*mean_of_ox*2*(rand(k,1)-0.5+1i*(rand(k,1)-0.5));

gamma_init = max(min(1.5, (1-2*log(n)*r/n)*(1/alpha)),1.2);
gamma_decay = 0.95;
tol = 1e-8;
max_iter = 200;
proj = false;
eta = 0.5;

[x,err,timer] = HSNLD(obs,n,r,M,alpha,eta,gamma_init,gamma_decay,proj,tol,max_iter);

recovery_err = norm(ox-x)/norm(ox)

figure
plot(1:length(timer(1:end,1)),err(1:end,1),'*-');
grid on
legend('HSNLD');
ylabel('Relative Error');
xlabel('Iteration Number');
set(gca,'YScale', 'log')

