class AlwaysDriftDetector(DriftDetector):
    def check_drift(self, data_window):
        return True