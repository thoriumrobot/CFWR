/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    @Positive
    public Super(int @MinLen(2) [] minlen2) {
        short __cfwr_item91 = null;

    @Positive
      this.minlen2 = minlen2;
    @Positive
    }
    @Positive
  }

  // :: error: (field.invariant.not.subtype)
    @Positive
  class InvalidSub extends Super {
    @Positive
    public InvalidSub() {
    @Positive
      super(new int[] {1, 2});
    @Positive
    }
    @Positive
  }

    @Positive
  class ValidSub extends Super {
    @Positive
    public final int[] validSubField;

    @Positive
    public ValidSub(int[] validSubField) {
    @Positive
      super(new int[] {1, 2, 3, 4});
    @Positive
      this.validSubField = validSubField;
    @Positive
        protected int __cfwr_compute70(float __cfwr_p0, Long __cfwr_p1, long __cfwr_p2) {
        return true;
        return -492;
    }
    private static Integer __cfwr_helper873() {
        try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 2; __cfwr_i62++) {
            if ((true ^ 43.30) || (null / (827 - -81.86f))) {
            if (true || false) {
            return 57.15;
        }
        }
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        if (false || (-491L % 's')) {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 8; __cfwr_i23++) {
            while (true) {
            if (true && ((false | null) * false)) {
            return false;
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}
