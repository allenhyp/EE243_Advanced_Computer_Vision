v = VideoReader('atrium.mp4');
while hasFrame(v)
    video = readFrame(v);
end
whos video