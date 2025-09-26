/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        for (int __cfwr_i34 = 0; __cfwr_i34 < 7; __cfwr_i34++) {
            return null;
        }

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
      private static int __cfwr_func18() {
        while (true) {
            while (false) {
            Integer __cfwr_elem32 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i96 = 0; __cfwr_i96 < 2; __cfwr_i96++) {
            if (true || false) {
            boolean __cfwr_obj76 = false;
        }
        }
        try {
            Integer __cfwr_temp34 = null;
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        return ((-401 | 47.63) >> 54.34);
    }
    Boolean __cfwr_temp38(double __cfwr_p0, Boolean __cfwr_p1) {
        Boolean __cfwr_obj8 = null;
        return null;
    }
    protected String __cfwr_compute545(Character __cfwr_p0) {
        if ((true & -40.70f) && false) {
            return null;
        }
        return "test73";
    }
}
