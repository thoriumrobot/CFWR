/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        Boolean __cfwr_data58 = null;

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
      public static char __cfwr_proc97() {
        try {
            if (false || (55.74f << 782L)) {
            if ((6.93 & false) && true) {
            while ((null >> 'p')) {
            if ((null | ('5' << -311)) && (null >> 507L)) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 8; __cfwr_i57++) {
            try {
            double __cfwr_temp71 = 55.62;
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        return 'U';
    }
}
