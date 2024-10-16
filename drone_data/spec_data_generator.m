function [P, p, frequencies] = spec_data_generator(freqrange, timerange, signal2noise, midfreq, sigfreq, freqpix, timepix)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
buff = ones(freqrange,timerange);
%noisesig = awgn(zeros(1,freqrange*timerange),noise);

freqorder = [];

noisefloor = -270 + 10*randn;

carry = db2pow(noisefloor)*((abs((awgn(buff,3)))+0.01).^0.75);
bin = zeros(freqrange,timerange);

n = 0;
prefreq = midfreq;

while n < floor(timerange/timepix)
    freq = randi([midfreq-sigfreq/2+1,midfreq+sigfreq/2-freqpix+1]);
    rand1 = rand(1);
    rand2 = rand(1);


    if (prefreq - 2*freqpix > freq) || (freq > prefreq + 2*freqpix)
        carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = ((db2pow(signal2noise)+1) * carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix)).* abs(awgn(ones(freqpix,timepix),20));
        bin(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = 1;
        freqorder(n+1) = freq;


        if rand1 < 0.1 && n > 0
            if prefreq < freq

                carry(round(prefreq/4+3*freq/4):freq-1,n*timepix+1) = (db2pow(signal2noise/2)+1) * carry(round(prefreq/4+3*freq/4):freq-1,n*timepix+1);
            else

                carry(freq + freqpix : round(prefreq/4+3*freq/4),n*timepix+1) = (db2pow(signal2noise/2)+1) * carry(freq + freqpix : round(prefreq/4+3*freq/4),n*timepix+1);
            end

        end
        if rand2 < 0.2


            m = 0;

            while m < 10

                if 0 < freq-m
                    if rand1 < 0.1 && prefreq < freq
                        carry(freq-m-1,n*timepix+2:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq-m-1,n*timepix+2:(n+1)*timepix);
                    else
                        carry(freq-m-1,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq-m-1,n*timepix+1:(n+1)*timepix);
                    end
                end
                if freq+m+1 <= freqrange
                    if rand1 < 0.1 && prefreq > freq
                        carry(freq+freqpix+m,n*timepix+2:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq+freqpix+m,n*timepix+2:(n+1)*timepix);
                    else
                        carry(freq+freqpix+m,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq+freqpix+m,n*timepix+1:(n+1)*timepix);
                    end
                end
                m = m + 1;
            end
        end
    else
        if prefreq < freq
            freq = prefreq + 2*freqpix;
            carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)+1) * carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix).* abs(awgn(ones(freqpix,timepix),20));
            bin(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = 1;
            freqorder(n+1) = freq;
        else
            freq = prefreq - 2*freqpix ;
            carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)+1) * carry(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix).* abs(awgn(ones(freqpix,timepix),20));
            bin(freq:freq+freqpix-1,n*timepix+1:(n+1)*timepix) = 1;
            freqorder(n+1) = freq;
        end

        if rand1 < 0.1 && n > 0
            if prefreq < freq

                carry(round(prefreq/4+3*freq/4):freq-1,n*timepix+1) = (db2pow(signal2noise/2)+1) * carry(round(prefreq/4+3*freq/4):freq-1,n*timepix+1);
            else

                carry(freq + freqpix : round(prefreq/4+3*freq/4),n*timepix+1) = (db2pow(signal2noise/2)+1) * carry(freq + freqpix : round(prefreq/4+3*freq/4),n*timepix+1);
            end

        end
        if rand2 < 0.2
            m = 0;
            while m < 10

                if 0 < freq-m
                    if rand1 < 0.1 && prefreq < freq
                        carry(freq-m-1,n*timepix+2:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq-m-1,n*timepix+2:(n+1)*timepix);
                    else
                        carry(freq-m-1,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^4)/5+1) * carry(freq-m-1,n*timepix+1:(n+1)*timepix);
                    end
                end
                if freq+m+1 <= freqrange
                    if rand1 < 0.1 && prefreq > freq
                        carry(freq+freqpix+m,n*timepix+2:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^5)/4+1) * carry(freq+freqpix+m,n*timepix+2:(n+1)*timepix);
                    else
                        carry(freq+freqpix+m,n*timepix+1:(n+1)*timepix) = (db2pow(signal2noise)*((1-m/10)^5)/4+1) * carry(freq+freqpix+m,n*timepix+1:(n+1)*timepix);
                    end
                end
                m = m + 1;
            end
        end
    end

    prefreq = freq;
    n = n +1;
end    

p = bin;
P = pow2db(carry);
frequencies = freqorder;

end




