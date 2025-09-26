/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        return null;

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
      static float __cfwr_util456() {
        return 743;
        Double __cfwr_result39 = null;
        if (false || true) {
            try {
            while (false) {
            while (true) {
            int __cfwr_result70 = ('1' ^ null);
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        }
        return -82.97f;
    }
    static double __cfwr_temp524(String __cfwr_p0, char __cfwr_p1, Float __cfwr_p2) {
        while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e7) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i6 = 0; __cfwr_i6 < 10; __cfwr_i6++) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 2; __cfwr_i30++) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 10; __cfwr_i12++) {
            while ((315 | (false ^ 55.81f))) {
            while (false) {
            if ((null + true) || false) {
            String __cfwr_elem66 = "world57";
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return ('k' << -272L);
    }
}
