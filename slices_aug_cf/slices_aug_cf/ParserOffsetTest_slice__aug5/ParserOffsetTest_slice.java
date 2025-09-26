/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        Float __cfwr_obj33 
        if (true || true) {
            while (false) {
            Boolean __cfwr_val12 = null;
            break; // Prevent infinite loops
        }
        }
= null;

    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "1") int k = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction5(String[] a, int i) {
    if (1 - i < a.length) {
      // :: error: (assignment)
      @IndexFor("a") int j = i;
    }
  }

  @SuppressWarnings("lowerbound")
  public void subtraction6(String[] a, int i, int j) {
    if (i - j < a.length - 1) {
      @IndexFor("a") int k = i - j;
      // :: error: (assignment)
      @IndexFor("a") int k1 = i;
    }
  }

  public void multiplication1(String[] a, int i, @Positive int j) {
    if ((i * j) < (a.length + j)) {
      // :: error: (assignment)
      @IndexFor("a") int k = i;
      // :: error: (assignment)
      @IndexFor("a") int k1 = j;
    }
  }

  public void multiplication2(String @ArrayLen(5) [] a, @IntVal(-2) int i, @IntVal(20) int j) {
    if ((i * j) < (a.length - 20)) {
      @LTLengthOf("a") int k1 = i;
      // :: error: (assignment)
      @LTLengthOf(value = "a", offset = "20") int k2 = i;
      // :: error: (assignment)
      @LTLengthOf("a") int k3 = j;
    }
      public static char __cfwr_temp891(int __cfwr_p0, float __cfwr_p1) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 7; __cfwr_i94++) {
            if (false || (29.22f | (-82.56 * -65.00f))) {
            int __cfwr_entry90 = -193;
        }
        }
        if (false && (826 | true)) {
            try {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 4; __cfwr_i56++) {
            if ((258 % (-59.84f - -73.11f)) || true) {
            short __cfwr_var1 = null;
        }
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        }
        return '7';
    }
    public Character __cfwr_helper329(String __cfwr_p0, Double __cfwr_p1, float __cfwr_p2) {
        Character __cfwr_var17 = null;
        if (false && ((null % -700L) & null)) {
            return ((false >> true) / -204L);
        }
        for (int __cfwr_i36 = 0; __cfwr_i36 < 5; __cfwr_i36++) {
            byte __cfwr_elem23 = null;
        }
        return null;
        return null;
    }
}
