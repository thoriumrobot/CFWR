/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        for (int __cfwr_i72 = 0; __cfwr_i72 < 5; __cfwr_i72++) {
            for (int __cfwr_i54 = 0; __cfwr_i54 < 2; __cfwr_i54++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
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
      public static Boolean __cfwr_calc922(float __cfwr_p0) {
        char __cfwr_elem16 = 'l';
        Boolean __cfwr_obj26 = null;
        return null;
        return '7';
        return null;
    }
    public static boolean __cfwr_handle979(Float __cfwr_p0) {
        Float __cfwr_temp85 = null;
        Object __cfwr_obj71 = null;
        return ('p' * '2');
    }
}
