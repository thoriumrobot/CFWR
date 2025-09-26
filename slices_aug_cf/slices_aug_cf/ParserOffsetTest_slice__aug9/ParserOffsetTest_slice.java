/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
        try {
            boolean __cfwr_item52 = ((null ^ 88.78) + false);
        } catch (Exception __cfwr_e95) {
            // ignore
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
      public Long __cfwr_aux10(String __cfwr_p0) {
        while (true) {
            if (false && (68.89f % null)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 6; __cfwr_i6++) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 7; __cfwr_i29++) {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 2; __cfwr_i65++) {
            return null;
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i87 = 0; __cfwr_i87 < 1; __cfwr_i87++) {
            if (false && ((true & null) >> (null + null))) {
            try {
            try {
            while ((13.39 - '6')) {
            return (false + (-20.37 | 942));
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }
        }
        while ((null & 58.23)) {
            return (null << null);
            break; // Prevent infinite loops
        }
        while (true) {
            for (int __cfwr_i56 = 0; __cfwr_i56 < 1; __cfwr_i56++) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 4; __cfwr_i16++) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 3; __cfwr_i68++) {
            String __cfwr_entry32 = "data78";
        }
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
