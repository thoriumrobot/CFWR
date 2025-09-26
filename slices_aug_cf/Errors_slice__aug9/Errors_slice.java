/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        if (true || (('B' % true) ^ -15.17f)) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 9; __cfwr_i8++) {
            try {
            while (false) {
            boolean __cfwr_elem23 = true;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        }
        }

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
      private static Integer __cfwr_temp136() {
        while (false) {
            if ((50L * 'r') || true) {
            Boolean __cfwr_elem82 = null;
        }
            break; // Prevent infinite loops
        }
        while (true) {
            double __cfwr_node94 = 62.38;
            break; // Prevent infinite loops
        }
        return null;
        return false;
        return null;
    }
}
