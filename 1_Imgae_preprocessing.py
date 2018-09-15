from Module.data import Preprocessing

dir_path = 'Data/Images/'
L_list = [1.0, 2.0, 4.0, 8.0]
save_path = 'Data/Images.h5'
save_path2 = 'Data/Images_sampled.h5'
n_samples = 400

model = Preprocessing(dir_path)
model.readData()
model.noiseImageGenerating(L_list=L_list, file_name=save_path)
model.sampling(L_list=L_list, load_path=save_path, save_path=save_path2, n_samples=n_samples)