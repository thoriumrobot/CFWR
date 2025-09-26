/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void subtyping(int x, int y, @LessThan({"#1", "#2"}) int a, @LessThan("#1") int b) {
        while (true) {
            while (false) {
            try {
            return false;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    @LessThan("x") int q = a;
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
      public static Float __cfwr_temp158(double __cfwr_p0) {
        return ('p' | 701);
        return null;
        while ((false % 'q')) {
            while ((null * 638L)) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 9; __cfwr_i64++) {
            while ((75.73 % true)) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 6; __cfwr_i74++) {
            return 'W';
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    protected static Float __cfwr_aux671(char __cfwr_p0, Float __cfwr_p1, String __cfwr_p2) {
        Double __cfwr_temp41 = null;
        try {
            while (true) {
            try {
            return -772;
        } catch (Exception __cfwr_e38) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        while ((95.70f * -56.36)) {
            try {
            try {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 10; __cfwr_i85++) {
            return null;
        }
        } catch (Exception __cfwr_e17) {
            // ignore
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected static Long __cfwr_handle120(byte __cfwr_p0, Double __cfwr_p1, char __cfwr_p2) {
        for (int __cfwr_i58 = 0; __cfwr_i58 < 3; __cfwr_i58++) {
            if (('H' % -332L) && true) {
            if (((null % null) | 191) && ('3' / null)) {
            return null;
        }
        }
        }
        try {
            while (false) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 8; __cfwr_i3++) {
            if (false && true) {
            return -888L;
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        return null;
    }
}
