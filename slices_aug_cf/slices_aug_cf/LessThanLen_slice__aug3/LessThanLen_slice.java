/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        if (false && (-30.00f >> 'Y')) {
            int __cfwr_obj92 = 221;
        }

    int[] longer = new int[shorter.length * 1];
    // :: error: (assignment)
    @LTLengthOf("longer") int x = shorter.length;
    @LTEqLengthOf("longer") int y = shorter.length;
  }

  public static void m5(int[] shorter) {
    // :: error: (array.length.negative)
    int[] longer = new int[shorter.length * -1];
    // :: error: (assignment)
    @LTLengthOf("longer") int x = shorter.length;
    // :: error: (assignment)
    @LTEqLengthOf("longer") int y = shorter.length;
  }

  public static void m6(int @MinLen(1) [] shorter) {
    int[] longer = new int[4 * shorter.length];
    // TODO: enable when https://github.com/kelloggm/checker-framework/issues/211 is fixed
    // // :: error: (assignment)
    // @LTLengthOf("longer") int x = shorter.length;
    @LTEqLengthOf("longer") int y = shorter.length;
      private static char __cfwr_handle768(int __cfwr_p0, double __cfwr_p1) {
        if (true && true) {
            Float __cfwr_elem11 = null;
        }
        try {
            while (((-76.38 * 434L) - null)) {
            while (true) {
            float __cfwr_entry99 = -82.15f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        while (true) {
            if (((null & 45.94f) >> (-58.32f >> null)) && false) {
            try {
            boolean __cfwr_result23 = (-713 % null);
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return 'u';
    }
}
