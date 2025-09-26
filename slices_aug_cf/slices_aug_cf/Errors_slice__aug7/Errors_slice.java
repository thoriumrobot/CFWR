/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        if (true && (309L * true)) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 4; __cfwr_i74++) {
            Double __cfwr_obj72 = null;
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
    int a = arr
        double __cfwr_elem53 = (null & (null << 'n'));
[n1p];

    // :: error: (array.access.unsafe.low)
    int b = arr[u];

    int c = arr[nn];
    int d = arr[p];
      protected byte __cfwr_func839(short __cfwr_p0) {
        if (false && true) {
            try {
            while (false) {
            return ('i' >> (165 | -622));
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
        return null;
    }
    public static long __cfwr_func387(long __cfwr_p0, boolean __cfwr_p1) {
        try {
            return '3';
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        float __cfwr_result77 = 67.49f;
        return 289L;
    }
    protected Double __cfwr_func340(short __cfwr_p0) {
        if (false || false) {
            return null;
        }
        return null;
    }
}
