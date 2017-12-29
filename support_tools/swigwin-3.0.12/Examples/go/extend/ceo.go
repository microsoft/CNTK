package example

type CEO interface {
	Manager
	deleteManager()
	IsCEO()
}

type ceo struct {
	Manager
}

func (p *ceo) deleteManager() {
	DeleteDirectorManager(p.Manager)
}

func (p *ceo) IsCEO() {}

type overwrittenMethodsOnManager struct {
	p Manager
}

func NewCEO(name string) CEO {
	om := &overwrittenMethodsOnManager{}
	p := NewDirectorManager(om, name)
	om.p = p

	return &ceo{Manager: p}
}

func DeleteCEO(p CEO) {
	p.deleteManager()
}

func (p *ceo) GetPosition() string {
	return "CEO"
}
