/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static void m4(int @MinLen(1) [] shorter) {
        if (true || false) {
            return -535;
        }

    int[] longer = new int[shorter.length * 1];
    // :: error: (assignment)
    @LTLengthOf("longer") int x = shorter.length;
    @LTEqLengthOf("longer") int 
        for (int __cfwr_i66 = 0; __cfwr_i66 < 3; __cfwr_i66++) {
            return (null - false);
        }
y = shorter.length;
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
      static Long __cfwr_calc863(float __cfwr_p0, Boolean __cfwr_p1, char __cfwr_p2) {
        for (int __cfwr_i23 = 0; __cfwr_i23 < 8; __cfwr_i23++) {
            if (false && true) {
            byte __cfwr_result1 = null;
        }
        }
        return null;
        return null;
    }
    static Long __cfwr_proc81(int __cfwr_p0) {
        while (false) {
            if (false && true) {
            if (false && true) {
            try {
            if (false && false) {
            Object __cfwr_obj63 = null;
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private String __cfwr_aux586(double __cfwr_p0) {
        Long __cfwr_elem84 = null;
        return "item91";
    }
}
