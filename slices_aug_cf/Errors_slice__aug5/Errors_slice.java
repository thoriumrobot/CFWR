/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        return null;

    int[] arr = new int[5];

    // unsafe
    @GTENegativeOne int n1p = -1;
    @LowerBoundUnknown int u = -10;

    // safe
    @NonNegative int nn = 0;
    @Positive int p = 1;

    // :: error: (array.access.unsafe.low)
    int a = arr[n1p];

    // :: error: (array.access.unsafe.low)
    int b = arr[u];

    int c = arr[nn];
    int d = arr[p];
      protected Boolean __cfwr_aux103(Character __cfwr_p0, int __cfwr_p1, Character __cfwr_p2) {
        double __cfwr_var90 = 20.74;
        for (int __cfwr_i4 = 0; __cfwr_i4 < 2; __cfwr_i4++) {
            if (true || (null ^ null)) {
            try {
            return null;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        }
        }
        try {
            try {
            if (true && (-38.58f - (-9.45 % 895L))) {
            while (false) {
            return (false >> 155);
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        for (int __cfwr_i13 = 0; __cfwr_i13 < 5; __cfwr_i13++) {
            if (true || false) {
            return -43.63f;
        }
        }
        return null;
    }
}
