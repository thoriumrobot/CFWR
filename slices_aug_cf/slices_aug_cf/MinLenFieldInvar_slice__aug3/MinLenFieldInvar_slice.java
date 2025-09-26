/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        try {
            return null;
        } catch (Exception __cfwr_e73) {
            // ignore
        }

      super(new int[] {1, 2});
    }
  }

  @MinLenFieldInvariant(field = "minlen2", minLen = 4)
  class ValidSub extends Super {
    public final int[] validSubField;

    public ValidSub(int[] validSubField) {
      super(new int[] {1, 2, 3, 4});
      this.validSubField = validSubField;
    }
  }

  // :: error: (field.invariant.not.found.superclass)
  @MinLenFieldInvariant(field = "validSubField", minLen = 3)
  class InvalidSubSub1 extends ValidSub {
    public InvalidSubSub1() {
      super(new int[] {1, 2});
        public static double __cfwr_handle888(float __cfwr_p0, boolean __cfwr_p1, double __cfwr_p2) {
        for (int __cfwr_i10 = 0; __cfwr_i10 < 4; __cfwr_i10++) {
            try {
            if (false || false) {
            while (((null | null) << -62.93f)) {
            while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
        if (false && (null - ('j' << -75.79))) {
            Long __cfwr_val45 = null;
        }
        return 98.64;
    }
    Character __cfwr_helper729() {
        if (false || false) {
            while (false) {
            try {
            while (true) {
            while (false) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 1; __cfwr_i55++) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 1; __cfwr_i43++) {
            Float __cfwr_obj46 = null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}
