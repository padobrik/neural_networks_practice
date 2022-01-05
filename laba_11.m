clear;
close all hidden;
%{
drawing letter 'B' 10x10
 0 - black
 1 - white
%}

letter = [1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1;
          1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 1;
          1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
          1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1;
          1 1 1 1 0 0 1 1 1 1 1 1 1 1 0 0 0 1 1 1;
          1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1;
          1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1];

%{
Hopfield network requires matrix with -1 for black
and 1 for white, let's replace all zeros to -1
and plot a letter:
%}

letter(letter==0) = -1;

% Training our network
net = newhop(letter);
T1 = letter;
[Y1,Pf1,Af1] = sim(net, 20, [], T1);

% Adding noise to initial image and plotting it
T2 = imnoise(letter, 'salt & pepper', 0.10);

figure(1),
subplot(121), imshow(letter), title('Original letter');
subplot(122), imshow(T2), title('Noisy letter');

% Recovering noised image point by point
for i = 1:20
    Ai1 = {T2(:, i)};
    Ai2(:, i) = cell2mat(Ai1);
    [Y2, Pf2, Af2] = sim(net, [1 20], [], Ai1);
    T3(:, i) = Y2{1};
end

% Compare noised and recovered images
figure(2), 
subplot(121), imshow(T2), title('Noised image'); hold on;
subplot(122), imshow(T3); title('Recovered image'); hold on;