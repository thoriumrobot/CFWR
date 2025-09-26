/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        if (false || false) {
            return null;
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
      protected static int __cfwr_handle499(short __cfwr_p0) {
        try {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 6; __cfwr_i53++) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 7; __cfwr_i26++) {
            Boolean __cfwr_obj30 = null;
        }
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        return (false % ('a' - 730));
        Float __cfwr_elem81 = null;
        return -653;
    }
    Boolean __cfwr_proc224(Integer __cfwr_p0, short __cfwr_p1) {
        for (int __cfwr_i83 = 0; __cfwr_i83 < 9; __cfwr_i83++) {
            long __cfwr_node70 = -804L;
        }
        return null;
    }
    protected short __cfwr_compute754(int __cfwr_p0, char __cfwr_p1, Long __cfwr_p2) {
        try {
            try {
            try {
            long __cfwr_var79 = -560L;
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        return null;
    }
}
