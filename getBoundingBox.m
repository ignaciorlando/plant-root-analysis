
function [i_min, i_max, j_min, j_max] = getBoundingBox(mask)

    sumoverrows = sum(mask,1);
    [j_min] = find(sumoverrows,1,'first');
    [j_max] = find(sumoverrows,1,'last');
    
    sumoovercolumns = sum(mask,2);
    [i_min] = find(sumoovercolumns,1,'first');
    [i_max] = find(sumoovercolumns,1,'last');


end