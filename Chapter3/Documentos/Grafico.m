close all;
clear all;
clc;

sz=100

xideal=[0,1]
yideal=[0,1]

% Valor t?cnico
x=[0.68 0.41 0.68];

% Valor econ?mico
y=[0.65 0.62 0.6];

figure('Name','Evaluaci?n T?cnica - Econ?mica')
scatter(x(1),y(1),sz,'b','^','filled')
grid on;
hold on;
scatter(x(2),y(2),sz,'g','s','filled')
scatter(x(3),y(3),sz,'r','d','filled')
plot(xideal,yideal,'k')
axis([0 1 0 1])
ylabel('Valor Econ?mico');
xlabel('Valor T?cnico');
%title('Evaluaci?n T?cnico-econ?mica')
legend('Soluci?n 1', 'Soluci?n 2', 'Soluci?n 3')