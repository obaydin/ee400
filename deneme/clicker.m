x = 50;
y = 50;
plot(x,y,'gs', 'linewidth', 20);
axis([0 100 0 100]);


temp = 1;
while temp
    mouse_position = get(gca,'CurrentPoint');
    x = mouse_position(1,1)
    y = mouse_position(1,2)
    plot(x,y,'gs', 'linewidth', 20);
    axis([0 100 0 100]);
    pause(0.1)
end