clear;
close all hidden;

n = 12;
maze = -50*ones(n,n);
for i=1:(n-3)*n
    maze(randi([1,n]),randi([1,n]))=1;
end

maze(1,1) = 1;
maze(n,n) = 10;
disp(maze)
n=length(maze);
figure
imagesc(maze)
colormap(winter)

for i=1:n
    for j=1:n
        if maze(i,j)==min(maze)
        text(j,i,'X','HorizontalAlignment','center')
        end
    end
end

text(1,1,'START','HorizontalAlignment','center')
text(n,n,'GOAL','HorizontalAlignment','center')
axis off

Goal=n*n;
fprintf('Goal State is: %d',Goal)
reward=zeros(n*n);

for i=1:Goal
    reward(i,:)=reshape(maze',1,Goal);
end

for i=1:Goal
    for j=1:Goal
        if j~=i-n && j~=i+n && j~=i-1 && j~=i+1 && j~=i+n+1 && j~=i+n-1 ...
        && j~=i-n+1 && j~=i-n-1
            reward(i,j)=-Inf;
        end
    end
end

for i=1:n:Goal
    for j=1:i+n
        if j==i+n-1 || j==i-1 || j==i-n-1
            reward(i,j)=-Inf;
            reward(j,i)=-Inf;
        end
    end
end

q = randn(size(reward));
gamma = 0.9;
alpha = 0.2;
maxItr = 50;

for i=1:maxItr
% Starting from start position
    cs=1;
% Repeat until Goal state is reached
    while(1)
% possible actions for the chosen state
        n_actions = find(reward(cs,:)>0);
% choose an action at random and set it as the next state
        ns = n_actions(randi([1 length(n_actions)],1,1));
% find all the possible actions for the selected state
        n_actions = find(reward(ns,:)>=0);
% find the maximum q-value i.e, next state with best action
        max_q = 0;
            for j=1:length(n_actions)
            max_q = max(max_q,q(ns,n_actions(j)));
            end
% Update q-values as per Bellman's equation
        q(cs,ns)=reward(cs,ns)+gamma*max_q;
% Check whether the episode has completed i.e Goal has been reached
        if(cs == Goal)
            break;
        end
% Set current state as next state
        cs=ns;
    end
end

start = 1;
move = 0;
path = start;

while(move~=Goal)
    [~,move]=max(q(start,:));
% Deleting chances of getting stuck in small loops (upto order of 4)
    if ismember(move,path)
        [~,x]=sort(q(start,:),'descend');
        move=x(2);
            if ismember(move,path)
                [~,x]=sort(q(start,:),'descend');
                move=x(3);
                    if ismember(move,path)
                        [~,x]=sort(q(start,:),'descend');
                        move=x(4);
                            if ismember(move,path)
                                [~,x]=sort(q(start,:),'descend');
                                move=x(5);
                            end
                    end
            end
    end

% Appending next action/move to the path
    path=[path,move];
    start=move;
end

fprintf('Final path: %s',num2str(path))
fprintf('Total steps: %d',length(path))

% reproducing path to matrix path
pmat=zeros(n,n);
[q, r]=quorem(sym(path),sym(n));
q=double(q);r=double(r);
q(r~=0)=q(r~=0)+1;r(r==0)=n;

for i=1:length(q)
    pmat(q(i),r(i))=50;
end

figure
imagesc(pmat)
colormap(white)

for i=1:n
    for j=1:n
        if maze(i,j)==min(maze)
text(j,i,'X','HorizontalAlignment','center')
end
if pmat(i,j)==50
text(j,i,'\bullet','Color','red','FontSize',28)
end
end
end
text(1,1,'START','HorizontalAlignment','right')
text(n,n,'GOAL','HorizontalAlignment','right')
hold on
imagesc(maze,'AlphaData',0.2)
colormap(winter)
hold off
axis off