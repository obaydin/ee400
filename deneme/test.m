snrvals = [8;16];
pixvals = [4;20];
n = 0;

while n < 2

    n = n +1;
    for snr = snrvals
        %[n snr]
        for pix = pixvals
            
            midfreq = randi([400 6000]);
            sigfreq = 2*randi([50 400]);
            [P,p,freqs] = spec_data_generator(6400,1000,snr,midfreq,sigfreq,pix(1),pix(2));
    
            frequencies = freqs;
            %saveP = append("C:\Users\STAJYER\Desktop\data_easy\",'snr_',int2str(snr),'\','pixs_',num2str(pix(1)),'_',num2str(pix(2)),'\','specs');

            %mkdir(saveP)
            %save(append(saveP,'\snr_',int2str(snr),'pixes',num2str(pix(1)),'_',num2str(pix(2)),'_v',num2str(n),'.mat'),"P")
            %savep = append("C:\Users\STAJYER\Desktop\data_easy\",'snr_',int2str(snr),'\','pixs_',num2str(pix(1)),'_',num2str(pix(2)),'\','binary');
            %mkdir(savep)
            %save(append(savep,'\snr_',int2str(snr),'pixes',num2str(pix(1)),'_',num2str(pix(2)),'_v',num2str(n),'bnr.mat'),'p')
        end
    end
    
end
 