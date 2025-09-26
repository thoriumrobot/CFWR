/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        while ((-135 - (null / -1.28))) {
            while (((89 + true) + 432L)) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

    return indexOf(array, target, 0, array.length);
  }

  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  GuavaPrimitives(short @MinLen(1) [] array) {
    this(array, 0, array.length);
  }

  @SuppressWarnings(
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
  )
  GuavaPrimitives(
      short @MinLen(1) [] array,
      @IndexFor("#1") @LessThan("#3") int start,
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    this.array = array;
    this.start = start;
    this.end = end;
  }

  public @Positive @LTLengthOf(
      value = {"this", "array"},
      offset = {"-1", "start - 1"}) int
      size() { // INDEX: Annotation on a public method refers to private member.
    return end - start;
  }

  public boolean isEmpty() {
    return false;
  }

  public Short get(@IndexFor("this") int index) {
    return array[start + index];
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  public Short set(@IndexFor("this") int index, Short element) {
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    array[start + index] = element;
    return oldValue;
  }

  @SuppressWarnings("index") // needs https://github.com/kelloggm/checker-framework/issues/229
  public List<Short> subList(
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    int size = size();
    if (fromIndex == toIndex) {
      return Collections.emptyList();
    }
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder(size() * 6);
    builder.append('[').append(array[start]);
    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    return builder.append(']').toString();
      static Integer __cfwr_compute628() {
        for (int __cfwr_i90 = 0; __cfwr_i90 < 8; __cfwr_i90++) {
            while (true) {
            for (int __cfwr_i80 = 0; __cfwr_i80 < 2; __cfwr_i80++) {
            short __cfwr_val2 = (('3' / 809) & -77.84f);
        }
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i10 = 0; __cfwr_i10 < 5; __cfwr_i10++) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 10; __cfwr_i89++) {
            while (((33.65 & null) * (null ^ true))) {
            String __cfwr_result92 = "data85";
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        Character __cfwr_val28 = null;
        return null;
    }
    static Character __cfwr_util349(Integer __cfwr_p0) {
        return null;
        try {
            boolean __cfwr_obj47 = false;
        } catch (Exception __cfwr_e80) {
            // ignore
        }
        try {
            if (true && ((-57L - -94.28f) * -73L)) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 2; __cfwr_i34++) {
            try {
            Long __cfwr_var28 = null;
        } catch (Exception __cfwr_e65) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        return null;
    }
}
