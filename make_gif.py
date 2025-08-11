import imageio, glob
imgs = sorted(glob.glob("samples/snap_*.jpg"))
frames = [imageio.imread(f) for f in imgs]
if frames:
    imageio.mimsave("samples/demo.gif", frames, duration=0.6)
    print("samples/demo.gif created")
else:
    print("No frames found")
