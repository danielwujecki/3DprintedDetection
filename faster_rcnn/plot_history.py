import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config


conf = Config()
record_df = pd.read_csv(conf.history_path)
r_epochs = len(record_df)

plt.style.use('ggplot')

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['loss_rpn_cls'], 'r')
plt.title('loss_rpn_cls')

plt.subplot(2, 2, 2)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['loss_rpn_regr'], 'r')
plt.title('loss_rpn_regr')

plt.subplot(2, 2, 3)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['loss_class_cls'], 'r')
plt.title('loss_class_cls')

plt.subplot(2, 2, 4)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['loss_class_regr'], 'r')
plt.title('loss_class_regr')

plt.savefig(conf.plot_path.split('.')[0] + '_losses.png')
plt.close()

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['mean_overlapping_bboxes'], 'r')
plt.title('mean_overlapping_bboxes')

plt.subplot(2, 2, 2)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['class_acc'], 'r')
plt.title('class_acc')

plt.subplot(2, 2, 3)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['curr_loss'], 'r')
plt.title('total_loss')

plt.subplot(2, 2, 4)
plt.plot(np.arange(r_epochs, dtype='int'),
         record_df['elapsed_time'] / 60, 'r')
plt.title('elapsed_time')

plt.savefig(conf.plot_path.split('.')[0] + '_general.png')
plt.close()
