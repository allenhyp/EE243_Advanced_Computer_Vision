function pred = translate_max(pred)
[mm, nn] = size(pred);
for ii=1:mm
    lm=-1;
    for jj=1:nn
        if pred(ii,jj)>lm
            lm=pred(ii,jj);
        end
    end
    for jj=1:nn
        if pred(ii,jj)==lm
            pred(ii,jj)=1;
        else
            pred(ii,jj)=0;
        end
    end
end
