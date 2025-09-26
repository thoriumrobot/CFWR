// simplified version of bug in plume-lib fixed by b5df093053146bcf32f485eea974e70cc854b407

public class OptionsToHtml {

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public void optionsToHtml(boolean includeClassDoc, Object... root_classes) {
	if (includeClassDoc) {
	    //:: error: (array.access.unsafe.high.constant)
	    javadocToHtml(root_classes[0]);
	}
    }

    /*@ public normal_behavior
@ requires true;
@ ensures true;
@*/
    public void javadocToHtml(Object o) { }
}
