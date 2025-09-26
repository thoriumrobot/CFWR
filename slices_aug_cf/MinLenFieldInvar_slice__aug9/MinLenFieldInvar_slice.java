/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
    public InvalidSub() {
        Object __cfwr_data3 = null;

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
        public Character __cfwr_func721() {
        while (true) {
            while (true) {
            short __cfwr_var96 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    private Boolean __cfwr_calc628() {
        for (int __cfwr_i4 = 0; __cfwr_i4 < 4; __cfwr_i4++) {
            while (false) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 7; __cfwr_i49++) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 4; __cfwr_i47++) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 9; __cfwr_i32++) {
            try {
            if (true || true) {
            return null;
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    private Integer __cfwr_process572(int __cfwr_p0, Long __cfwr_p1) {
        return (false / 'D');
        return null;
    }
}
