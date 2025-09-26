/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        if (true && false) {
            if (true && (-143L - '7')) {
            if (false && (84.45 | (false >> 'B'))) {
            return null;
        }
        }
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
      static boolean __cfwr_proc694() {
        for (int __cfwr_i7 = 0; __cfwr_i7 < 8; __cfwr_i7++) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        char __cfwr_val64 = 'Q';
        try {
            return 760L;
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        long __cfwr_val10 = 188L;
        return (null | null);
    }
    protected static Integer __cfwr_proc618(Double __cfwr_p0, Float __cfwr_p1) {
        byte __cfwr_item73 = null;
        return null;
    }
}
