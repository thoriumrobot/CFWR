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
      static Integer __cfwr_func983(short __cfwr_p0, long __cfwr_p1) {
        for (int __cfwr_i62 = 0; __cfwr_i62 < 6; __cfwr_i62++) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 8; __cfwr_i24++) {
            if ((-1.27f ^ (null ^ 446L)) || ((null % -54.40f) % true)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 7; __cfwr_i6++) {
            if (true && true) {
            if (true && (('0' & null) >> 232)) {
            try {
            if (false && (921L % (true % false))) {
            long __cfwr_data98 = ((-77.51f / 56.15f) + -75.32);
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
        }
        }
        }
        }
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 8; __cfwr_i34++) {
            return null;
        }
        return null;
    }
    private static Boolean __cfwr_proc138(Float __cfwr_p0, Float __cfwr_p1, float __cfwr_p2) {
        while (((-105L & 98.59f) | (36 ^ false))) {
            boolean __cfwr_var68 = false;
            break; // Prevent infinite loops
        }
        return null;
    }
    private Integer __cfwr_aux878(Boolean __cfwr_p0, byte __cfwr_p1, long __cfwr_p2) {
        long __cfwr_val41 = -296L;
        return null;
    }
}
