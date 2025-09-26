/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        if (true || true) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 3; __cfwr_i58++) {
            return null;
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
      private static Float __cfwr_temp877(float __cfwr_p0) {
        if ((null ^ null) || ((-80.02 - false) & null)) {
            try {
            return null;
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        try {
            float __cfwr_elem76 = -20.78f;
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        return null;
    }
}
