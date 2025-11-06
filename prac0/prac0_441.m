function prac0_441
% Nxx1 line code: transition at each bit start; 0 has mid-bit transition.

bits_str = '10010';
b = bits_str - '0';
L = 8;                     % samples/bit (>=4)
[s,t] = nxx1_encode(b,L);

figure('Name','Ex4.4.1 Nxx1');
stairs(t, s,'LineWidth',1.5); grid on;
xlabel('Time (bit periods)'); ylabel('Level');
title(sprintf('Nxx1 for bits %s (L=%d)', bits_str, L));
ylim([-1.5 1.5]);
end

function [y,t] = nxx1_encode(b,L)
lev = -1;                       % initial level
N = numel(b); y = zeros(1,N*L);
for k = 1:N
    lev = -lev;                 % transition at bit start
    y((k-1)*L + (1:L/2)) = lev;
    if b(k)==0, lev = -lev; end % 0: transition in middle; 1: no transition
    y((k-1)*L + (L/2+1:L)) = lev;
end
t = (0:numel(y)-1)/L;           % in bit periods
end
