import os


class STMWriter:
    def __init__(self, save_dir):
        os.makedirs(os.path.join(save_dir, 'save_train_details'), exist_ok=True)
        self.train_loss = os.path.join(save_dir, 'save_train_details', 'train_loss.txt')
        self.val_loss = os.path.join(save_dir, 'save_train_details', 'val_loss.txt')
        self.test_result = os.path.join(save_dir, 'save_train_details', 'test_result.txt')

    def write_train(self, msg):
        with open(self.train_loss, 'a') as f:
            f.write(msg + '\n')
            f.close()

    def write_val(self, msg):
        with open(self.val_loss, 'a') as f:
            f.write(msg + '\n')
            f.close()

    def write_test(self, msg):
        with open(self.test_result, 'a') as f:
            f.write(msg + '\n')
            f.close()
