from dependency import *
class Model:
    def __init__(self,Data,n):
        self.load(Data,n)

    def load(self,Data,n):
        data = []
        data_set = []
        data_set_labels = []
        for person in Data.males:
            data.append([person.flatten(),"male"])
        for person in Data.females:
            data.append([person.flatten(),"female"])
        random.seed(datetime.now().microsecond)
        random.shuffle(data)
        for tuple in data:
            data_set.append(tuple[0])
            data_set_labels.append(tuple[1])
        h,w=person.shape
        X_train, X_test, y_train, y_test = train_test_split(data_set, data_set_labels, test_size=0.1, random_state=datetime.now().second)
        n_components = n
        self.pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
        X_train_pca = self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        self.clf = LinearSVC()
        self.clf.fit(X_train_pca,y_train)
        s=self.clf.score(X_test_pca,y_test)
        print "Train Score ",s
        print "Model succesfully Built."

    def plot_gallery(images, titles, h, w, n_row=5, n_col=15):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
        plt.show()
