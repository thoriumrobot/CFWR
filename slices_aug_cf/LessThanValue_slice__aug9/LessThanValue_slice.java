/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        for (int __cfwr_i7 = 0; __cfwr_i7 < 1; __cfwr_i7++) {
            while (false) {
            Integer __cfwr_elem45 = null;
            break; // Prevent infinite loops
        }
        }

    @LessThan("x") 
        for (int __cfwr_i14 = 0; __cfwr_i14 < 1; __cfwr_i14++) {
            while (false) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 7; __cfwr_i97++) {
            try {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 3; __cfwr_i36++) {
            float __cfwr_temp16 = -14.53f;
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
int q = a;
    @LessThan({"x", "y"})
    // :: error: (assignment)
    int r = b;
  }

  public static boolean flag;

  void lub(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
    @LessThan("x") int r = flag ? a : b;
    @LessThan({"x", "y"})
    // :: error: (assignment)
    int s = flag ? a : b;
  }

  void transitive(int a, int b, int c) {
    if (a < b) {
      if (b < c) {
        // :: error: (assignment)
        @LessThan("c") int x = a;
      }
    }
  }

  void calls() {
    isLessThan(0, 1);
    isLessThanOrEqual(0, 0);
  }

  void isLessThan(@LessThan("#2") @NonNegative int start, int end) {
    @NonNegative int x = end - start - 1;
    @Positive int y = end - start;
  }

  @NonNegative int isLessThanOrEqual(@LessThan("#2 + 1") @NonNegative int start, int end) {
    return end - start;
  }

  public void setMaximumItemCount(int maximum) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    int count = getCount();
    if (count > maximum) {
      @Positive int y = count - maximum;
      @NonNegative int deleteIndex = count - maximum - 1;
    }
  }

  int getCount() {
    throw new RuntimeException();
  }

  void method(@NonNegative int m) {
    boolean[] has_modulus = new boolean[m];
    @LessThan("m") int x = foo(m);
    @IndexFor("has_modulus") int rem = foo(m);
  }

  @LessThan("#1") @NonNegative int foo(int in) {
    throw new RuntimeException();
  }

  void test(int maximum, int count) {
    if (maximum < 0) {
      throw new IllegalArgumentException("Negative 'maximum' argument.");
    }
    if (count > maximum) {
      int deleteIndex = count - maximum - 1;
      // TODO: shouldn't error
      // :: error: (argument)
      isLessThanOrEqual(0, deleteIndex);
    }
  }

  void count(int count) {
    if (count > 0) {
      if (count % 2 == 1) {

      } else {
        // TODO: improve value checker
        // :: error: (assignment)
        @IntRange(from = 0) int countDivMinus = count / 2 - 1;
        // Reasign to update the value in the store.
        countDivMinus = countDivMinus;
        // :: error: (argument)
        isLessThan(0, countDivMinus);
        isLessThanOrEqual(0, countDivMinus);
      }
    }
  }

  static @NonNegative @LessThan("#2 + 1") int expandedCapacity(
      @NonNegative int oldCapacity, @NonNegative int minCapacity) {
    if (minCapacity < 0) {
      throw new AssertionError("cannot store more than MAX_VALUE elements");
    }
    // careful of overflow!
    int newCapacity = oldCapacity + (oldCapacity >> 1) + 1; // expand by %50
    if (newCapacity < minCapacity) {
      newCapacity = Integer.highestOneBit(minCapacity - 1) << 1;
    }
    if (newCapacity < 0) {
      newCapacity = Integer.MAX_VALUE;
      // guaranteed to be >= newCapacity
    }
    // :: error: (return)
    return newCapacity;
      public static boolean __cfwr_util347(boolean __cfwr_p0, Integer __cfwr_p1, short __cfwr_p2) {
        if ((-287 << 290L) || false) {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 1; __cfwr_i50++) {
            while (true) {
            try {
            if (true || false) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            return 701L;
        }
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        for (int __cfwr_i63 = 0; __cfwr_i63 < 1; __cfwr_i63++) {
            try {
            return -819L;
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        }
        return (-67.22 | (324 << null));
    }
    public Object __cfwr_calc767(Boolean __cfwr_p0, float __cfwr_p1, Character __cfwr_p2) {
        return ((109 + -49.96f) >> -29.72);
        return null;
    }
}
