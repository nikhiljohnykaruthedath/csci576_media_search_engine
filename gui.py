import glob
import os
import time
import pickle
import code
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfilename
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from skimage.transform import resize
import matplotlib
import config
from video import Video
from video_player import VideoPlayer
from feature_extraction import extract_features
from feature_comparison import compare_features, rank_features, generate_plots, generate_plot, display_plot
import cv2

class VideoQueryGUI(ttk.Frame):
    FORCE_CREATE = False

    def __init__(self, master):
        super(VideoQueryGUI, self).__init__()
        self.master = master
        master.title("CSCI 576 Project")
        master.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        self.folders = [x[0] for x in os.walk(config.DB_VID_ROOT)][1:]
        self.query_scores = None

        self.create_frames()
        self.load_db_thread = threading.Thread(target=self.load_database, name='database loader')
        self.load_db_thread.start()

    def load_database(self):
        self.update_status('>>> Loading DB videos', clear=True)
        print('Started')
        print('=' * 80)
        print('Database video list')
        print('-' * 80)
        print('\n'.join(['%d. %s' % (i+1, f)
                         for (i, f) in enumerate(self.folders)]))
        print('=' * 80)

        self.db_vids = []
        for selected_folder in self.folders:
            self.update_status('>>> DB video selected: ' + selected_folder)
            pkl_path = glob.glob(os.path.join(selected_folder, '*.pkl'))
            if len(pkl_path) and not self.FORCE_CREATE:
                tic = time.time()
                self.update_status('>>> Loading pre-calculated features')
                with open(pkl_path[0], 'rb') as pkl_fp:
                    v = pickle.load(pkl_fp)
                self.update_status('>>> Done. Time taken: %0.4fs' % (time.time()-tic))
            else:
                tic = time.time()
                self.update_status('>>> Loading video')
                vid_path = selected_folder
                aud_path = glob.glob(os.path.join(selected_folder, '*.wav'))[0]
                v = Video(vid_path, aud_path)
                self.update_status('>>> Done. Time taken: %0.4fs' % (time.time()-tic))

                # Computing features
                tic = time.time()
                self.update_status('>>> Calculating video features')
                extract_features(v)
                self.update_status('>>> Calculated in %0.4fs' % (time.time()-tic))

                self.update_status('>>> Saving results to database')
                with open(os.path.join(selected_folder, '%s.pkl' % v.name), 'wb') as pkl_fp:
                    pickle.dump(v, pkl_fp)
            self.db_vids.append(v)
            self.update_status('>>> Saved results to database')

    def create_frames(self):
        # Top frame
        self.top_frame = tk.LabelFrame(self.master, text='', bg="#E9E9E9")
        self.top_frame.pack(side='top', expand=True, fill='both')

        self.load_query_button = ttk.Button(self.top_frame, text='Load Query', command=self.load_query_video)
        self.load_query_button.grid(row=0, column=0, padx=0, pady=0)

        self.find_matches_button = ttk.Button(self.top_frame, text='Find matches', command=self.run_match)
        self.find_matches_button.grid(row=1, column=0, padx=0, pady=0)

        self.match_list = tk.Listbox(self.top_frame, height=4, bd=0)
        self.yscroll = tk.Scrollbar(self.top_frame, orient=tk.VERTICAL)
        self.match_list['yscrollcommand'] = self.yscroll.set
        self.match_list.grid(row=0, column=1, rowspan=2, stick='wens')
        self.yscroll.grid(row=0, column=1, rowspan=2, sticky='nse')
        self.match_list.bind('<Double-Button-1>', self.poll_match_list)
        self.curr_selection = -1

        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=2)

        # Middle frame
        self.middle_frame = tk.LabelFrame(self.master, text='')
        self.middle_frame.pack(side='top', expand=True, fill='both')

        self.status_label_text = tk.StringVar()
        self.status_label_text.set('LOGS')
        self.status_label = tk.Label(self.middle_frame, textvar=self.status_label_text, justify=tk.LEFT, anchor='w', wraplength=700, bg="black", fg="#C1E0FD")
        self.status_label.grid(row=0, column=0, stick='nswe', columnspan=2)

        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.middle_frame.grid_columnconfigure(1, weight=1)

        self.query_player = VideoPlayer(self.middle_frame)
        self.query_player.grid(row=1, column=0, stick='nsw')
        self.db_player = VideoPlayer(self.middle_frame)
        self.db_player.grid(row=1, column=1, stick='nse')

        # Bottom frame
        self.bottom_frame = tk.LabelFrame(self.master, text='')
        self.bottom_frame.pack(side='top', expand=True, fill='both')

        self.match_info_label_text = tk.StringVar()
        # self.match_info_label_text.set('MATCH INFORMATION')
        self.match_info_label = tk.Label(self.bottom_frame, textvar=self.match_info_label_text, justify=tk.LEFT, anchor='e')
        self.match_info_label.grid(row=0, column=0, stick='nswe')

        image = cv2.imread("/Users/nikhiljohny/Documents/_CSCI576Project/codebase/master/graph_place_holder.png")
        width = 365
        height = 280
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        self.panelA = tk.Label(self.bottom_frame, image=image, width=365, height=280)
        self.panelA.image = image
        self.panelA.grid(row=0, column=1, stick='nse', padx=0, pady=0)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

    def load_query_video(self):
        selected_folder = None
        self.master.update()
        selected_folder = askdirectory(
            initialdir=config.QUERY_VID_ROOT, title='Select query folder')
        # selected_folder = askopenfilename(
        #     initialdir=config.QUERY_VID_ROOT, title='Select query folder')
        self.update_status('>>> Selected: ' + selected_folder)

        if selected_folder == '':
            return

        self.query_loader = threading.Thread(
            target=self.load_query, args=(selected_folder, ), name='query_loader')
        self.query_loader.start()

        # while self.query_loader.is_alive():
        #     self.update_idletasks()

    def load_query(self, selected_folder=None):
        self.update_status('>>> Loading query video', clear=True)
        if selected_folder == None:
            selected_folder = os.path.join(
                config.QUERY_VID_ROOT, 'subclip_traffic')
            # selected_folder = askdirectory(
            #     initialdir=config.QUERY_VID_ROOT, title='Select query folder')
            # print(selected_folder)
        self.update_status('>>> Query video selected: '+ selected_folder)
        # print('Selected '+selected_folder)

        pkl_path = glob.glob(os.path.join(selected_folder, 'query_scores.pkl'))

        # print(pkl_path)

        if len(pkl_path) and not self.FORCE_CREATE:
            vid_pkl_path = [pth for pth in glob.glob(os.path.join(
                selected_folder, '*.pkl')) if not os.path.basename(pth).startswith('query_scores')]
            tic = time.time()
            self.update_status('>>> Loading pre-calculated metrics')
            # print('Loading pre-calculated comparison metrics')
            with open(pkl_path[0], 'rb') as pkl_fp:
                self.query_scores = pickle.load(pkl_fp)
            with open(vid_pkl_path[0], 'rb') as pkl_fp:
                self.query_vid = pickle.load(pkl_fp)
            self.update_status('>>> Done. Time taken: %0.4fs' % (time.time()-tic))
        else:
            pkl_path = [pth for pth in glob.glob(os.path.join(
                selected_folder, '*.pkl')) if not os.path.basename(pth).startswith('query_scores')]

            if len(pkl_path) and not self.FORCE_CREATE:
                tic = time.time()
                self.update_status('>>> Loading pre-calculated features')
                with open(pkl_path[0], 'rb') as pkl_fp:
                    self.query_vid = pickle.load(pkl_fp)
                self.update_status('>>> Done. Time taken: %0.4fs' % (time.time()-tic))
            else:
                # Loading query video
                tic = time.time()
                vid_path = selected_folder
                aud_path = glob.glob(os.path.join(selected_folder, '*.wav'))[0]
                self.update_status('>>> Loading query video %s' %
                                   os.path.basename(vid_path))
                self.query_vid = Video(vid_path, aud_path)

                # Computing features
                tic = time.time()
                self.update_status('>>> Calculating video features')
                # print('Calculating video features')
                extract_features(self.query_vid)
                self.update_status('>>> Calculated in %0.4fs' % (time.time()-tic))
                # print('Calculated in %0.4fs' % (time.time()-tic))

                self.update_status('>>> Creating pickle file for query video')
                with open(os.path.join(selected_folder, '%s.pkl' % self.query_vid.name), 'wb') as pkl_fp:
                    pickle.dump(self.query_vid, pkl_fp)

            self.query_scores = {}
            for i, db_vid in enumerate(self.db_vids):
                self.update_status('>>> Comparing features with %s' % db_vid.name)
                tic = time.time()
                self.query_scores[db_vid.name] = compare_features(
                    self.query_vid, db_vid)
                self.update_status('>>> Feature comparison completed in %0.4fs' %
                                   (time.time()-tic))

            self.update_status('>>> Saving results to database')
            with open(os.path.join(selected_folder, 'query_scores.pkl'), 'wb') as pkl_fp:
                pickle.dump(self.query_scores, pkl_fp)
            self.update_status('>>> Saved results to database')

        self.query_player.load_video(self.query_vid)

    def run_match(self):
        if self.query_scores is None:
            self.update_status('>>> No query video selected', clear=True)
            return
        self.update_status('>>> Running query in database', clear=True)
        self.final_ranks = rank_features(self.query_scores)
        display_plot(self.final_ranks, "graph", "/Users/nikhiljohny/Documents/_CSCI576Project/codebase/master/graph")
        generate_plots(self.final_ranks, "temp", "/Users/nikhiljohny/Documents/_CSCI576Project/codebase/master/temp")

        start_fr = np.argmax(self.final_ranks[0][3])
        query_vid_len = len(self.db_vids[0].frames) - len(self.final_ranks[0][3]) + 1

        self.update_match_info('')
        self.update_match_info('Best Match Video: {}'.format(self.final_ranks[0][0]))
        self.update_match_info('Start Frame: {}'.format(start_fr))
        self.update_match_info('End Frame: {}'.format(start_fr + query_vid_len))
        start_time = round(start_fr/config.FRAME_RATE)
        self.update_match_info('Start Time: {}'.format(start_time))
        end_time = round(float(start_fr + query_vid_len) / config.FRAME_RATE)
        self.update_match_info('End Time: {}'.format(end_time))

        self.update_status('>>> Final scores computed')
        image = cv2.imread("/Users/nikhiljohny/Documents/_CSCI576Project/codebase/master/graph/graph_final.png")
        width = 365
        height = 280
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.panelA.configure(image=image)
        self.panelA.image = image

        self.update_status('>>> Ranked list of matches generated')
        self.match_list.delete(0, tk.END)
        i = 0
        matchlist = [x[0] for x in self.final_ranks]
        for match in matchlist:
            final_score = str(round(self.final_ranks[i][1], 4))
            match = '{} | {}'.format(match, final_score)
            self.match_list.insert(tk.END, match)
            i = i + 1

        self.poll_match_list()

    def poll_match_list(self, event=None):
        current = self.match_list.curselection()[0]
        # print(current)
        if current != self.curr_selection:
            self.update_status('>>> Video selected: ' +
                                self.final_ranks[current][0])
            curr_video = self.find_matching_db_vid(
                self.final_ranks[current][0])
            self.db_player.load_video(curr_video)
            plot = generate_plots(self.final_ranks[current])
            
            self.corr_plot = resize(
                plot, (100, 356, 3), preserve_range=True).astype('uint8')
            self.curr_selection = current
        self.master.after(250, self.poll_match_list)

    def find_matching_db_vid(self, vidname):
        for vid in self.db_vids:
            if vid.name == vidname:
                return vid
        else:
            return None

    def update_status(self, text, clear=False):
        if clear:
            status_text = '\t\t\t\t         LOGS\n\n%s' % text
        else:
            status_text = '%s\n%s' % (self.status_label_text.get(), text)
        lines = status_text.split('\n')
        if len(lines) < 6:
            status_text = '\n'.join(lines + ['']*(6-len(lines)))
        elif len(lines) > 6:
            status_text = '\n'.join([lines[0]]+lines[-5:])
        self.status_label_text.set(status_text)

    def update_match_info(self, text, clear=False):
        if clear:
            match_info_text = 'MATCH INFORMATION:\n%s' % text
        else:
            match_info_text = '%s\n%s' % (self.match_info_label_text.get(), text)
        lines = match_info_text.split('\n')
        if len(lines) < 6:
            match_info_text = '\n'.join(lines + ['']*(6-len(lines)))
        elif len(lines) > 6:
            match_info_text = '\n'.join([lines[0]]+lines[-5:])
        self.match_info_label_text.set(match_info_text)

    def onClose(self):
        self.query_player.onClose()
        self.db_player.onClose()
        self.master.quit()

if __name__ == '__main__':
    root = tk.Tk()
    app = VideoQueryGUI(root)
    root.mainloop()
    root.destroy()