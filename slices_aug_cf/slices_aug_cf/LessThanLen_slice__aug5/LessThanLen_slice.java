/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        while (false) {
            if (((false << -863L) * null) || ((null + 68.21) | 350)) {
            String __cfwr_data70 = "world31";
        }
            break; // Prevent infinite loops
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
      protected static int __cfwr_proc600(double __cfwr_p0) {
        try {
            Integer __cfwr_obj86 = null;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        for (int __cfwr_i23 = 0; __cfwr_i23 < 1; __cfwr_i23++) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 10; __cfwr_i79++) {
            try {
            Object __cfwr_entry64 = null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        }
        try {
            try {
            while (true) {
            if (false && false) {
            try {
            try {
            return null;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return 950;
    }
}
